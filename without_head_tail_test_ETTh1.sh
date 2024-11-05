export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_without_head_tail


# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --limited_lambda_mode 2 \
  --without_head_tail 1 \
  --is_training 1 \
  --mask_rate 0.125 \
  --requires_grad True \
  --train_mode 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id iTransformer_ETTh1_0.125_96_96_J_lambda_stu_ReLu_without_head_tail \
  --model $model_name \
  --data ETTh1 \
  --features M \
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
  --des 'Exp' \
  --itr 1 
