export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer



# Mask rate 0.375
python -u mix_run.py \
  --task_name long_term_forecast \
  --limited_lambda_mode 1 \
  --is_training 1 \
  --mask_rate 0.375 \
  --requires_grad True \
  --train_mode 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Crossformer_ECL_0.375_96_96_J_lambda_stu_clamp \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 
  
# Mask rate 0.5
python -u mix_run.py \
  --task_name long_term_forecast \
  --limited_lambda_mode 1 \
  --is_training 1 \
  --mask_rate 0.5 \
  --requires_grad True \
  --train_mode 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Crossformer_ECL_0.5_96_96_J_lambda_stu_clamp \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 