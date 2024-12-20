export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

# Mask rate 0.125
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.625 \
  --_lambda 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_0.625_96_96_J_PatchTST_lambda_0_not_stu \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 32 \
  --itr 1 \
  --top_k 5 

# Mask rate 0.25
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.75 \
  --_lambda 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_0.75_96_96_J_PatchTST_lambda_0_not_stu \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 32 \
  --itr 1 \
  --top_k 5 

# Mask rate 0.125
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.625 \
  --_lambda 0.5 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_0.625_96_96_J_PatchTST_lambda_05_not_stu \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 32 \
  --itr 1 \
  --top_k 5 

# Mask rate 0.25
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.75 \
  --_lambda 0.5 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_0.75_96_96_J_PatchTST_lambda_05_not_stu \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 32 \
  --itr 1 \
  --top_k 5 
  
# Mask rate 0.125
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.625 \
  --_lambda 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_0.625_96_96_J_PatchTST_lambda_1_not_stu \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 32 \
  --itr 1 \
  --top_k 5 

# Mask rate 0.25
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.75 \
  --_lambda 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_0.75_96_96_J_PatchTST_lambda_1_not_stu \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 32 \
  --itr 1 \
  --top_k 5 