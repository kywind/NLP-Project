export CUDA_VISIBLE_DEVICES=0

python3 cli.py \
  --data_dir ../FewGLUE_32dev/BoolQ \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v2 \
  --task_name boolq \
  --output_dir ../output_xxl_noadv/boolq \
  --pet_per_gpu_eval_batch_size 8 \
  --pet_per_gpu_train_batch_size 8 \
  --pet_gradient_accumulation_steps 1 \
  --pet_max_seq_length 256 \
  --pet_max_steps 150 \
  --pattern_ids 1 \
  --overwrite_output_dir \
  --embed_size 128 \
  --prompt_encoder_type mlp \
  --adv_method_name InputSpecificAdversarial \
  --learning_rate 1e-4 \
  --seed 42 \
  --pet_repetitions 1 \
  --do_eval \
  --do_eval_adv \
  # --do_train \
  # --calibrate \
  # --do_train_adv \
  # --adv_ratio 0.2 \