#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=Informer

# Mask rate 0.25
python -u mix_run.py \
  --task_name long_term_forecast \
  --interpolate "linear" \
  --linear_nearest_only 1 \
  --train_mode 2 \
  --is_training 1 \
  --mask_rate 0.25 \
  --root_path "./dataset/ETT-small/" \
  --data_path "ETTh1.csv" \
  --model_id "ETTh1_0.25_96_96_R_linear_imputate" \
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
  --itr 1 \
  --top_k 5 

