#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "no" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.625 \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.625_96_96_R_no_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512\
  --d_ff 512\
  --des 'Exp' \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "no" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.75 \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.75_96_96_R_no_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512\
  --d_ff 512\
  --des 'Exp' \
  --itr 1  

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.625 \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.625_96_96_R_linear_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512\
  --d_ff 512\
  --des 'Exp' \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.75 \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.75_96_96_R_linear_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512\
  --d_ff 512\
  --des 'Exp' \
  --itr 1  

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "nearest" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.625 \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.625_96_96_R_nearest_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512\
  --d_ff 512\
  --des 'Exp' \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "nearest" \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.75 \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.75_96_96_R_nearest_imputation" \
  --model "$model_name" \
  --data "custom" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512\
  --d_ff 512\
  --des 'Exp' \
  --itr 1  