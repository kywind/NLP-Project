from torch.utils.data import dataset
import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.dataset import Dataset
from tqdm import trange, tqdm
from transformers import InputExample

import log
from pet import preprocessor
from data_utils.task_processors import task_helpers
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, RobertaForMaskedLM, BertConfig, BertTokenizer, RobertaConfig, \
    RobertaTokenizer, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig
from pet.utils import InputFeatures, DictDataset, distillation_loss, exact_match, ListDataset
from pet.wrapper import Adversarial, TransformerModelWrapper
import copy


class RandomPerturbationAdversarial(Adversarial):
    def __init__(self, adv_config):
        super().__init__(adv_config)
        self.wrapper = None
        self.attack1 = None
        self.attack2 = None
        self.no_training = True
    
    def get_adversarial(self, input_example: List[InputExample], attack1=None, attack2=None)->Dict:
        for example_instance in input_example:
            text_a, text_b = example_instance.text_a, example_instance.text_b
            for _ in range(3):
                a = np.random.randint(0, len(text_a))
                b = np.random.randint(0, len(text_a))
                text_a = text_a[:a] + text_a[b] + text_a[a+1:]
                a = np.random.randint(0, len(text_b))
                b = np.random.randint(0, len(text_b))
                text_b = text_b[:a] + text_b[b] + text_b[a+1:]
            example_instance.text_a, example_instance.text_b = text_a, text_b
        return {'adv_example': input_example}
    
    def train_adversarial_step(self, input_example: List[InputExample], wrapper: TransformerModelWrapper) -> Dict:
        return super().train_adversarial_step(input_example, wrapper)



class HotFlipAdversarial(Adversarial):
    def __init__(self, adv_config):
        super().__init__(adv_config)
        self.config = adv_config
        self.wrapper = None
        self.no_training = False
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        self.vocab_size = 30000
        self.len_prefix = 5
        self.k = 4
        self.attack1 = 'the the the the the'
        self.attack2 = 'the the the the they'
        candidates = []
        for i in range(self.vocab_size):
            if i in self.tokenizer.all_special_ids:
                continue
            if len(self.tokenizer.tokenize(self.tokenizer.convert_ids_to_tokens(i))) == 1:
                candidates.append(i)
        self.candidates = torch.tensor(candidates)
        
        # print(self.candidates.shape)
        # print(self.tokenizer.tokenize(self.initialization))
        # print(self.token_ids)
        # raise Exception

    def get_adversarial(self, input_example: List[InputExample], attack1, attack2=None)->Dict:
        for e in input_example:
            e.text_a, e.text_b = attack1 + ' ' + e.text_a, attack2 + ' ' + e.text_b
            # print(e.text_a)
            # print(e.text_b)
        return {'adv_example': input_example}


    def train_adversarial_step(self, input_example: List[InputExample], wrapper: TransformerModelWrapper) -> Dict:
        
        len_input = int(len(input_example) / 2)

        for _ in range(3): # epoch
            adv_example = self.get_adversarial(input_example[:len_input], self.attack1, self.attack2)['adv_example']
            adv_dataset = wrapper.generate_dataset(adv_example)
            adv_batch_size = len(adv_example)
            adv_dataloader = DataLoader(adv_dataset, batch_size=adv_batch_size)
            
            for batch in adv_dataloader: # just one batch
                self.optim.zero_grad()
                batch = {k: t.cuda() for k, t in batch.items()}
                labels = batch['labels']

                logits = wrapper.task_helper.eval_step(batch) if wrapper.task_helper else None
                if logits is None:
                    logits = self.get_eval_function()(wrapper)(batch)
                prediction_scores = logits.float().cuda()

                adv_loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(wrapper.config.label_list)), labels.view(-1))
                adv_loss_total = adv_loss.mean()   

                wrapper.raw_embeds.retain_grad()
                adv_loss_total.backward()
                embed_grad = wrapper.raw_embeds.grad.clone().detach()
                a_flag = batch['a_flag']
                b_flag = batch['b_flag']

                adv_grad = torch.zeros((a_flag.shape[0], self.len_prefix * 2, embed_grad.shape[2])).to(embed_grad.device)  # 2 (bsz) * 3 (length) * 128 (embedding dim)

                for i in range(a_flag.shape[0]):
                    idx_a = (a_flag[i] == 1).nonzero()[:self.len_prefix].reshape(-1)
                    la = idx_a.shape[0]
                    adv_grad[i, :la, :] = embed_grad[i, idx_a, :]

                    idx_b = (b_flag[i] == 1).nonzero()[:self.len_prefix].reshape(-1)
                    lb = idx_b.shape[0]
                    adv_grad[i, self.len_prefix:self.len_prefix+lb, :] = embed_grad[i, idx_b, :]

                model = wrapper.model.module if hasattr(self.model, 'module') else wrapper.model
                # if wrapper.config.model_type == "albert":
                all_embeds = model.model.albert.embeddings.word_embeddings(self.candidates.to(embed_grad.device))
                attack_id = self.tokenizer.encode(self.tokenizer.tokenize(self.attack1))[1:-1] + \
                            self.tokenizer.encode(self.tokenizer.tokenize(self.attack2))[1:-1]
                # print(self.attack)
                # print(self.tokenizer.tokenize(self.attack))
                # print(attack_id)
                if len(attack_id) != self.len_prefix * 2:
                    id1 = self.tokenizer.encode(self.tokenizer.tokenize(self.attack1))[1:-1]
                    id2 = self.tokenizer.encode(self.tokenizer.tokenize(self.attack2))[1:-1]
                    while len(id1) < self.len_prefix:
                        id1 = id1 + self.tokenizer.encode(self.tokenizer.tokenize('the'))[1:-1]
                    while len(id2) < self.len_prefix:
                        id2 = id2 + self.tokenizer.encode(self.tokenizer.tokenize('the'))[1:-1]
                    id1 = id1[:self.len_prefix]
                    id2 = id2[:self.len_prefix]
                    attack_id = id1 + id2
                assert len(attack_id) == self.len_prefix * 2

                cur_id = torch.tensor(attack_id).to(embed_grad.device)
                cur_embeds = model.model.albert.embeddings.word_embeddings(cur_id)
                diff_embeds = all_embeds.unsqueeze(0).expand(cur_embeds.shape[0], -1, -1) - cur_embeds.unsqueeze(1).expand(-1, all_embeds.shape[0], -1)
                adv_grad = adv_grad.mean(dim=0)

                diff_embeds /= torch.sum(diff_embeds * diff_embeds, axis=-1, keepdim=True)  # discourage large embed change

                adv_logits = torch.einsum('ijk,ik->ij', diff_embeds, adv_grad)  # 3 * 25570 (vocab size)
                adv_topk = torch.topk(adv_logits, self.k-1, dim=-1)[1]  # 3 * 2 (k-1)
                adv_topk = self.candidates[adv_topk.reshape(-1)].reshape(adv_topk.shape).to(embed_grad.device)
                adv_topk = torch.cat((adv_topk, cur_id[:, None]), dim=-1)  # 3 * 3 (k)

                # beam search, beam width = 1
                beam_best_attack_id = cur_id.clone()
                for pos in range(adv_topk.shape[0]):
                    best_loss = 0
                    beam_attack_id = beam_best_attack_id.clone()

                    # decide which attack id is best for position pos
                    for choice in range(self.k):
                        # if best_loss < 2:
                        #     continue
                        beam_attack_id[pos] = adv_topk[pos, choice]

                        beam_adv = self.tokenizer.convert_ids_to_tokens(beam_attack_id)
                        beam_attack1 = self.tokenizer.convert_tokens_to_string(beam_adv[:self.len_prefix])
                        beam_attack2 = self.tokenizer.convert_tokens_to_string(beam_adv[self.len_prefix:])
                        beam_example = self.get_adversarial(input_example[len_input:], beam_attack1, beam_attack2)['adv_example']
                        beam_dataset = wrapper.generate_dataset(beam_example)
                        beam_batch_size = len(beam_example)
                        beam_dataloader = DataLoader(beam_dataset, batch_size=beam_batch_size)

                        for batch in beam_dataloader: # just one batch
                            with torch.no_grad():
                                batch = {k: t.cuda() for k, t in batch.items()}
                                labels = batch['labels']
                                logits = wrapper.task_helper.eval_step(batch) if wrapper.task_helper else None
                                if logits is None:
                                    logits = self.get_eval_function()(wrapper)(batch)
                                prediction_scores = logits.float().cuda()
                                loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(wrapper.config.label_list)), labels.view(-1))
                                # print(pos, choice, loss, best_loss, beam_best_attack_id, beam_attack_id)
                                if loss > best_loss:  # temporary best attack id
                                    best_loss = loss
                                    beam_best_attack_id = beam_attack_id.clone()

                new_adv = self.tokenizer.convert_ids_to_tokens(beam_best_attack_id)
                self.attack1 = self.tokenizer.convert_tokens_to_string(new_adv[:self.len_prefix])
                self.attack2 = self.tokenizer.convert_tokens_to_string(new_adv[self.len_prefix:])


                # print(self.attack1)
                # print(self.attack2)
                # print(best_loss)

        return {'adv_loss_total': adv_loss_total}



class InputSpecificAdversarial(Adversarial):
    def __init__(self, adv_config):
        super().__init__(adv_config)
        self.config = adv_config
        self.wrapper = None
        self.no_training = True
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
        self.vocab_size = 30000
        self.len_prefix = 3
        self.k = 4
        self.attack1 = 'the the the'  # default, updated through training
        self.attack2 = 'the the the'
        candidates = []
        for i in range(self.vocab_size):
            if i in self.tokenizer.all_special_ids:
                continue
            if len(self.tokenizer.tokenize(self.tokenizer.convert_ids_to_tokens(i))) == 1:
                candidates.append(i)
        self.candidates = torch.tensor(candidates)


    def get_adversarial(self, input_example: List[InputExample], attack1_nouse=None, attack2_nouse=None)->Dict:
        for e in tqdm(input_example, desc="Generating attack"):
            attack1, attack2 = self.get_attack(e)
            e.text_a, e.text_b = attack1 + ' ' + e.text_a, attack2 + ' ' + e.text_b
        return {'adv_example': input_example}

    
    def train_adversarial_step(self, input_example: List[InputExample], wrapper: TransformerModelWrapper) -> Dict:
        return super().train_adversarial_step(input_example, wrapper)


    def get_given_adversarial(self, input_example: List[InputExample], attack1, attack2=None):
        input_example_copy = copy.deepcopy(input_example)
        for e in input_example_copy:
            e.text_a, e.text_b = attack1 + ' ' + e.text_a, attack2 + ' ' + e.text_b
        return {'adv_example': input_example_copy}


    def get_attack(self, input_example: InputExample):
        self.attack1 = 'the the the'
        self.attack2 = 'the the the'
        for _ in range(10): # epoch
            adv_example = self.get_given_adversarial([input_example], self.attack1, self.attack2)['adv_example']
            adv_dataset = self.wrapper.generate_dataset(adv_example)
            adv_batch_size = len(adv_example)
            assert adv_batch_size == 1
            adv_dataloader = DataLoader(adv_dataset, batch_size=adv_batch_size)
            
            for batch in adv_dataloader: # just one batch
                self.optim.zero_grad()
                batch = {k: t.cuda() for k, t in batch.items()}
                labels = batch['labels']

                logits = self.wrapper.task_helper.eval_step(batch) if self.wrapper.task_helper else None
                if logits is None:
                    logits = self.get_eval_function()(self.wrapper)(batch)
                prediction_scores = logits.float().cuda()

                adv_loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.wrapper.config.label_list)), labels.view(-1))
                adv_loss_total = adv_loss.mean()   

                self.wrapper.raw_embeds.retain_grad()
                adv_loss_total.backward()
                embed_grad = self.wrapper.raw_embeds.grad.clone().detach()
                a_flag = batch['a_flag']
                b_flag = batch['b_flag']

                adv_grad = torch.zeros((a_flag.shape[0], self.len_prefix * 2, embed_grad.shape[2])).to(embed_grad.device)  # 2 (bsz) * 3 (length) * 128 (embedding dim)

                for i in range(a_flag.shape[0]):
                    idx_a = (a_flag[i] == 1).nonzero()[:self.len_prefix].reshape(-1)
                    la = idx_a.shape[0]
                    adv_grad[i, :la, :] = embed_grad[i, idx_a, :]

                    idx_b = (b_flag[i] == 1).nonzero()[:self.len_prefix].reshape(-1)
                    lb = idx_b.shape[0]
                    adv_grad[i, self.len_prefix:self.len_prefix+lb, :] = embed_grad[i, idx_b, :]

                model = self.wrapper.model.module if hasattr(self.model, 'module') else self.wrapper.model
                # if self.wrapper.config.model_type == "albert":
                all_embeds = model.model.albert.embeddings.word_embeddings(self.candidates.to(embed_grad.device))
                attack_id = self.tokenizer.encode(self.tokenizer.tokenize(self.attack1))[1:-1] + \
                            self.tokenizer.encode(self.tokenizer.tokenize(self.attack2))[1:-1]
                # print(self.attack)
                # print(self.tokenizer.tokenize(self.attack))
                # print(attack_id)
                if len(attack_id) != self.len_prefix * 2:
                    id1 = self.tokenizer.encode(self.tokenizer.tokenize(self.attack1))[1:-1]
                    id2 = self.tokenizer.encode(self.tokenizer.tokenize(self.attack2))[1:-1]
                    while len(id1) < self.len_prefix:
                        id1 = id1 + self.tokenizer.encode(self.tokenizer.tokenize('the'))[1:-1]
                    while len(id2) < self.len_prefix:
                        id2 = id2 + self.tokenizer.encode(self.tokenizer.tokenize('the'))[1:-1]
                    id1 = id1[:self.len_prefix]
                    id2 = id2[:self.len_prefix]
                    attack_id = id1 + id2
                assert len(attack_id) == self.len_prefix * 2

                cur_id = torch.tensor(attack_id).to(embed_grad.device)
                cur_embeds = model.model.albert.embeddings.word_embeddings(cur_id)
                diff_embeds = all_embeds.unsqueeze(0).expand(cur_embeds.shape[0], -1, -1) - cur_embeds.unsqueeze(1).expand(-1, all_embeds.shape[0], -1)
                adv_grad = adv_grad.mean(dim=0)

                diff_embeds /= torch.sum(diff_embeds * diff_embeds, axis=-1, keepdim=True)  # discourage large embed change

                adv_logits = torch.einsum('ijk,ik->ij', diff_embeds, adv_grad)  # 3 * 25570 (vocab size)
                adv_topk = torch.topk(adv_logits, self.k-1, dim=-1)[1]  # 3 * 2 (k-1)
                adv_topk = self.candidates[adv_topk.reshape(-1)].reshape(adv_topk.shape).to(embed_grad.device)
                adv_topk = torch.cat((adv_topk, cur_id[:, None]), dim=-1)  # 3 * 3 (k)

                # beam search, beam width = 1
                beam_best_attack_id = cur_id.clone()
                for pos in range(adv_topk.shape[0]):
                    best_loss = 0
                    beam_attack_id = beam_best_attack_id.clone()

                    # decide which attack id is best for position pos
                    for choice in range(self.k):
                        # if best_loss < 2:
                        #     continue
                        beam_attack_id[pos] = adv_topk[pos, choice]

                        beam_adv = self.tokenizer.convert_ids_to_tokens(beam_attack_id)
                        beam_attack1 = self.tokenizer.convert_tokens_to_string(beam_adv[:self.len_prefix])
                        beam_attack2 = self.tokenizer.convert_tokens_to_string(beam_adv[self.len_prefix:])
                        beam_example = self.get_given_adversarial([input_example], beam_attack1, beam_attack2)['adv_example']
                        beam_dataset = self.wrapper.generate_dataset(beam_example)
                        beam_batch_size = len(beam_example)
                        beam_dataloader = DataLoader(beam_dataset, batch_size=beam_batch_size)

                        for batch in beam_dataloader: # just one batch
                            with torch.no_grad():
                                batch = {k: t.cuda() for k, t in batch.items()}
                                labels = batch['labels']
                                logits = self.wrapper.task_helper.eval_step(batch) if self.wrapper.task_helper else None
                                if logits is None:
                                    logits = self.get_eval_function()(self.wrapper)(batch)
                                prediction_scores = logits.float().cuda()
                                loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.wrapper.config.label_list)), labels.view(-1))
                                # print(pos, choice, loss, best_loss, beam_best_attack_id, beam_attack_id)
                                if loss > best_loss:  # temporary best attack id
                                    best_loss = loss
                                    beam_best_attack_id = beam_attack_id.clone()

                new_adv = self.tokenizer.convert_ids_to_tokens(beam_best_attack_id)
                self.attack1 = self.tokenizer.convert_tokens_to_string(new_adv[:self.len_prefix])
                self.attack2 = self.tokenizer.convert_tokens_to_string(new_adv[self.len_prefix:])

                # print(self.attack1)
                # print(self.attack2)
                # print(best_loss)

        return self.attack1, self.attack2



class FlipLabelAdversarial(Adversarial):  # only use at training  # not used
    def __init__(self, adv_config):
        super().__init__(adv_config)
        self.wrapper = None
        self.attack1 = None
        self.attack2 = None
        self.no_training = True
        self.flip_list = []
        self.train_size = 32
        self.cnt = 0
    
    def get_adversarial(self, input_example: List[InputExample], attack1=None, attack2=None)->Dict:
        # for idx in range(len(input_example)):
        #     if self.cnt in self.flip_list:
        #         print(input_example[idx].text_a)
        #         input_example[idx].label = 'True' if input_example[idx].label == 'False' else 'False'
        #     self.cnt += 1
        #     if self.cnt == self.train_size:
        #         self.cnt = 0
        return {'adv_example': input_example}
    
    def train_adversarial_step(self, input_example: List[InputExample], wrapper: TransformerModelWrapper) -> Dict:
        return super().train_adversarial_step(input_example, wrapper)



# perturb prefix --> data augmented training
# perturb word --> ? (flipda)
# perturb prompt template --> ? (nullprompt)
# perturb training example --> beam search style gradiend descent
# perturb initialization --> avoid spoiled initialization
