#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "no" \
  --train_mode 2 \
  --mask_rate 0.625 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.625_R_no_imputation_PatchTST \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "no" \
  --train_mode 2 \
  --mask_rate 0.75 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.75_R_no_imputation_PatchTST \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "linear" \
  --train_mode 2 \
  --mask_rate 0.625 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.625_R_linear_imputation_PatchTST \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "linear" \
  --train_mode 2 \
  --mask_rate 0.75 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.75_R_linear_imputation_PatchTST \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 
  
  # Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "nearest" \
  --train_mode 2 \
  --mask_rate 0.625 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.625_R_nearest_imputation_PatchTST \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "nearest" \
  --train_mode 2 \
  --mask_rate 0.75 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.75_R_nearest_imputation_PatchTST \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 