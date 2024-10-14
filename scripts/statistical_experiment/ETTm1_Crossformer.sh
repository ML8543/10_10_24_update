#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer

for mask_rate in  0.25 0.5 0.75
do
    python -u mix_run.py \
      --task_name long_term_forecast \
      --train_mode 0 \
      --is_training 1 \
      --do_statistic_exp 1 \
      --mask_rate $mask_rate \
      --imp_model_pt checkpoints/imputation_Crossformer_ETTm1_mask_${mask_rate}_Crossformer_ETTm1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
      --root_path "./dataset/ETT-small/" \
      --data_path "ETTm1.csv" \
      --model_id "Crossformer_ETTm1_${mask_rate}_96_96_I" \
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
done