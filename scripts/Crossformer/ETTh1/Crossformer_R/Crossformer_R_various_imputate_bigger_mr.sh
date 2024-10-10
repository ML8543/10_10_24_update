#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "no" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.625 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTh1.csv" \
  --model_id "Crossformer_ETTh1_0.625_96_96_R_no_imputate" \
  --model "$model_name" \
  --data "ETTh1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "no" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.75 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTh1.csv" \
  --model_id "Crossformer_ETTh1_0.75_96_96_R_no_imputate" \
  --model "$model_name" \
  --data "ETTh1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.625 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTh1.csv" \
  --model_id "Crossformer_ETTh1_0.625_96_96_R_linear_imputate" \
  --model "$model_name" \
  --data "ETTh1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.75 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTh1.csv" \
  --model_id "Crossformer_ETTh1_0.75_96_96_R_linear_imputate" \
  --model "$model_name" \
  --data "ETTh1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 
  
  # Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "nearest" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.625 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTh1.csv" \
  --model_id "Crossformer_ETTh1_0.625_96_96_R_nearest_imputate" \
  --model "$model_name" \
  --data "ETTh1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "nearest" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.75 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTh1.csv" \
  --model_id "Crossformer_ETTh1_0.75_96_96_R_nearest_imputate" \
  --model "$model_name" \
  --data "ETTh1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 