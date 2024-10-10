export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.125 \
  --_lambda 0.5 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_0.125_lambda_not_stu_PatchTST_J \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --d_model 128 \
  --d_ff 256 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16 \
  --n_heads 16 \
  --itr 1

# Mask rate 0.25
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.25 \
  --_lambda 0.5 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_0.25_lambda_not_stu_PatchTST_J \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --d_model 128 \
  --d_ff 256 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16 \
  --n_heads 16 \
  --itr 1

# Mask rate 0.375
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.375 \
  --_lambda 0.5 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_0.375_lambda_not_stu_PatchTST_J \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --d_model 128 \
  --d_ff 256 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16 \
  --n_heads 16 \
  --itr 1

# Mask rate 0.5
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.5 \
  --_lambda 0.5 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_0.5_lambda_not_stu_PatchTST_J \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --d_model 128 \
  --d_ff 256 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 16 \
  --n_heads 16 \
  --itr 1