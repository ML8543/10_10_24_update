#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

for mask_rate in  0.25 0.5 0.75
do
    python -u mix_run.py \
      --task_name long_term_forecast \
      --train_mode 0 \
      --is_training 1 \
      --do_statistic_exp 1 \
      --mask_rate $mask_rate \
      --imp_model_pt checkpoints/imputation_ETTh1_mask_${mask_rate}_PatchTST_ETTh1_ftM_sl96_ll0_pl0_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96_${mask_rate}_independent_PatchTST \
      --model $model_name \
      --data ETTh1 \
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
      --itr 1
done
