#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.125 \
  --root_path "./dataset/electricity/" \
  --data_path "electricity.csv" \
  --model_id "iTransformer_ECL_0.125_96_96_R_linear_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des "Exp" \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --top_k 5 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.25 \
  --root_path "./dataset/electricity/" \
  --data_path "electricity.csv" \
  --model_id "iTransformer_ECL_0.25_96_96_R_linear_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des "Exp" \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --top_k 5 

# Mask rate 0.375
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.375 \
  --root_path "./dataset/electricity/" \
  --data_path "electricity.csv" \
  --model_id "iTransformer_ECL_0.375_96_96_R_linear_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des "Exp" \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --top_k 5 

# Mask rate 0.5
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.5 \
  --root_path "./dataset/electricity/" \
  --data_path "electricity.csv" \
  --model_id "iTransformer_ECL_0.5_96_96_R_linear_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des "Exp" \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1 \
  --top_k 5 