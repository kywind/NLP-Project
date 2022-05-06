export CUDA_VISIBLE_DEVICES=0

python3 cli.py \
--data_dir ../FewGLUE_32dev/CB \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name cb \
--output_dir ../output/cb \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size 4 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 256 \
--pet_max_steps 250 \
--pattern_ids 1 \
--overwrite_output_dir \
--learning_rate 1e-4

# {'acc': 0.8928571428571429, 'f1-macro': 0.8678731870106534}
