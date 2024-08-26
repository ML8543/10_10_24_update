#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=Informer

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.125 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTm1.csv" \
  --model_id "ETTm1_0.125_96_96_R_linear_impute" \
  --model "$model_name" \
  --data "ETTm1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 \
  --top_k 5 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.25 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTm1.csv" \
  --model_id "ETTm1_0.25_96_96_R_linear_impute" \
  --model "$model_name" \
  --data "ETTm1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 \
  --top_k 5 

# Mask rate 0.375
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.375 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTm1.csv" \
  --model_id "ETTm1_0.375_96_96_R_linear_impute" \
  --model "$model_name" \
  --data "ETTm1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 \
  --top_k 5 

# Mask rate 0.5
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.5 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTm1.csv" \
  --model_id "ETTm1_0.5_96_96_R_linear_impute" \
  --model "$model_name" \
  --data "ETTm1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 \
  --top_k 5 