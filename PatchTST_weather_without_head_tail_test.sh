export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST_without_head_tail


# Mask rate 0.125
python -u mix_run.py \
  --task_name long_term_forecast \
  --limited_lambda_mode 2 \
  --without_head_tail 1 \
  --is_training 1 \
  --mask_rate 0.125 \
  --requires_grad True \
  --train_mode 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id patchTST_weather_96_96_0.125_J_lambda_stu_ReLu_without_head_tail \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 
