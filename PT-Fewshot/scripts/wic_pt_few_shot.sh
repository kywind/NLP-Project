export CUDA_VISIBLE_DEVICES=0

python3 cli.py \
--data_dir ../FewGLUE_32dev/WiC \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name wic \
--output_dir ../output/wic \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size 2 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 256 \
--pet_max_steps 3500 \
--pattern_ids 1
