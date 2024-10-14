#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=Informer

for mask_rate in  0.25 0.5 0.75
do
    python -u mix_run.py \
      --task_name long_term_forecast \
      --train_mode 0 \
      --is_training 1 \
      --do_statistic_exp 1 \
      --mask_rate $mask_rate \
      --imp_model_pt checkpoints/imputation_ECL_mask_${mask_rate}_Informer_custom_ftM_sl96_ll0_pl0_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
      --root_path "./dataset/electricity/" \
      --data_path "electricity.csv" \
      --model_id "Informer_ECL_${mask_rate}_96_96_I" \
      --model "$model_name" \
      --data "custom" \
      --features "M" \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des "Exp" \
      --itr 1 \
      --top_k 5 
done