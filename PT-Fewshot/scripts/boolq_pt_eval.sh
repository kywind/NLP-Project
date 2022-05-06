export CUDA_VISIBLE_DEVICES=0

python3 cli.py \
  --data_dir ../FewGLUE_32dev/BoolQ \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v2 \
  --task_name boolq \
  --output_dir ../output_xxl_noadv/boolq \
  --pet_per_gpu_eval_batch_size 4 \
  --pet_per_gpu_train_batch_size 2 \
  --pet_gradient_accumulation_steps 1 \
  --pet_max_seq_length 256 \
  --pet_max_steps 150 \
  --pattern_ids 1 \
  --overwrite_output_dir \
  --embed_size 128 \
  --prompt_encoder_type mlp \
  --learning_rate 1e-4 \
  --adv_method_name HotFlipAdversarial \
  --seed 1 \
  --do_eval \
  --do_eval_adv \
  --pet_repetition 3\
  #--calibrate