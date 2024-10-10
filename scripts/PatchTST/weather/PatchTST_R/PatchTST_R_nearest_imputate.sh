#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "nearest" \
  --train_mode 2 \
  --mask_rate 0.125 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.125_R_nearest_imputation_PatchTST \
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
  --mask_rate 0.25 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.25_R_nearest_imputation_PatchTST \
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

# Mask rate 0.375
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "nearest" \
  --train_mode 2 \
  --mask_rate 0.375 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.375_R_nearest_imputation_PatchTST \
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

# Mask rate 0.5
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "nearest" \
  --train_mode 2 \
  --mask_rate 0.5 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_0.5_R_nearest_imputation_PatchTST \
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