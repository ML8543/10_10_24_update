#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --is_training 1 \
  --mask_rate 0.125 \
  --imp_model_pt checkpoints/imputation_Crossformer_ETTm1_mask_0.125_Crossformer_ETTm1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTm1.csv" \
  --model_id "Crossformer_ETTm1_0.125_96_96_I" \
  --model "$model_name" \
  --data "ETTm1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --is_training 1 \
  --mask_rate 0.25 \
  --imp_model_pt checkpoints/imputation_Crossformer_ETTm1_mask_0.25_Crossformer_ETTm1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTm1.csv" \
  --model_id "Crossformer_ETTm1_0.25_96_96_I" \
  --model "$model_name" \
  --data "ETTm1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 

# Mask rate 0.375
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --is_training 1 \
  --mask_rate 0.375 \
  --imp_model_pt checkpoints/imputation_Crossformer_ETTm1_mask_0.375_Crossformer_ETTm1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTm1.csv" \
  --model_id "Crossformer_ETTm1_0.375_96_96_I" \
  --model "$model_name" \
  --data "ETTm1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 

# Mask rate 0.5
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --is_training 1 \
  --mask_rate 0.5 \
  --imp_model_pt checkpoints/imputation_Crossformer_ETTm1_mask_0.5_Crossformer_ETTm1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTm1.csv" \
  --model_id "Crossformer_ETTm1_0.5_96_96_I" \
  --model "$model_name" \
  --data "ETTm1" \
  --features "M" \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des "Exp" \
  --itr 1 