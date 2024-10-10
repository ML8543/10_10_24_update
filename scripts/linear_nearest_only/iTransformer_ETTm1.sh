#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

for interpolate in nearest linear
do
    for mask_rate in  0.125 0.25 0.375 0.5 0.625 0.75
    do
        python -u mix_run.py \
          --task_name long_term_forecast \
          --interpolate $interpolate \
          --linear_nearest_only 1 \
          --train_mode 2 \
          --is_training 1 \
          --mask_rate $mask_rate \
          --root_path "./dataset/ETT-small/" \
          --data_path "ETTm1.csv" \
          --model_id "ETTm1_${mask_rate}_96_96_R_${interpolate}_imputate" \
          --model "$model_name" \
          --data "ETTm1" \
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
          --d_model 128 \
          --d_ff 128 \
          --des "Exp" \
          --itr 1 
    done
done
