#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

for interpolate in nearest linear
do
    for mask_rate in  0.125 0.25 0.375 0.5 0.625 0.75
    do
        python -u mix_run.py \
          --task_name long_term_forecast \
          --interpolate $interpolate \
          --train_mode 2 \
          --linear_nearest_only 1 \
          --is_training 1 \
          --mask_rate $mask_rate \
          --root_path ./dataset/electricity/ \
          --data_path electricity.csv \
          --model_id "ECL_${mask_rate}_96_96_R_${interpolate}_imputate" \
          --model $model_name \
          --data custom \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len 96 \
          --e_layers 3 \
          --d_layers 1 \
          --factor 3 \
          --d_model 128 \
          --d_ff 256 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --batch_size 16 \
          --n_heads 16 \
          --itr 1
    done
done