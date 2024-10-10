#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --is_training 1 \
  --mask_rate 0.125 \
  --imp_model_pt checkpoints/imputation_weather_mask_0.125_iTransformer_custom_ftM_sl96_ll0_pl0_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.125_96_96_I" \
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
  --des "Exp" \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --is_training 1 \
  --mask_rate 0.25 \
  --imp_model_pt checkpoints/imputation_weather_mask_0.25_iTransformer_custom_ftM_sl96_ll0_pl0_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.25_96_96_I" \
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
  --des "Exp" \
  --itr 1 

# Mask rate 0.375
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --is_training 1 \
  --mask_rate 0.375 \
  --imp_model_pt checkpoints/imputation_weather_mask_0.375_iTransformer_custom_ftM_sl96_ll0_pl0_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.375_96_96_I" \
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
  --des "Exp" \
  --itr 1 

# Mask rate 0.5
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --is_training 1 \
  --mask_rate 0.5 \
  --imp_model_pt checkpoints/imputation_weather_mask_0.5_iTransformer_custom_ftM_sl96_ll0_pl0_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --root_path "./dataset/weather/" \
  --data_path "weather.csv" \
  --model_id "iTransformer_weather_0.5_96_96_I" \
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
  --des "Exp" \
  --itr 1 