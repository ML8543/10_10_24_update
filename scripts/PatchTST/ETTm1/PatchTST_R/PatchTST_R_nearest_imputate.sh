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
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_0.125_R_nearest_imputation_PatchTST \
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
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "nearest" \
  --train_mode 2 \
  --mask_rate 0.25 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_0.25_R_nearest_imputation_PatchTST \
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
  --itr 1 

# Mask rate 0.375
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "nearest" \
  --train_mode 2 \
  --mask_rate 0.375 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_0.375_R_nearest_imputation_PatchTST \
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
  --itr 1 

# Mask rate 0.5
python -u mix_run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --interpolate "nearest" \
  --train_mode 2 \
  --mask_rate 0.5 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_0.5_R_nearest_imputation_PatchTST \
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
  --itr 1 