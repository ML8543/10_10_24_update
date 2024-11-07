export CUDA_VISIBLE_DEVICES=0

model_name=Informer

# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --limited_lambda_mode 2 \
  --is_training 1 \
  --mask_rate 0.125 \
  --requires_grad True \
  --train_mode 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id Informer_ETTm1_0.125_96_96_J_lambda_stu_ReLu \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 
