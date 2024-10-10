export CUDA_VISIBLE_DEVICES=0

model_name=Crossformer

# Mask rate 0.125
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.125 \
  --_lambda 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Crossformer_weather_0.125_96_96_J_lambda1 \
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
  --top_k 5 \
  --des 'Exp' \
  --itr 1 

# Mask rate 0.25
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.25 \
  --_lambda 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Crossformer_weather_0.25_96_96_J_lambda1 \
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
  --top_k 5 \
  --des 'Exp' \
  --itr 1 

# Mask rate 0.375
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.375 \
  --_lambda 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Crossformer_weather_0.375_96_96_J_lambda1 \
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
  --top_k 5 \
  --des 'Exp' \
  --itr 1 

# Mask rate 0.5
python -u mix_run_me.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --mask_rate 0.5 \
  --_lambda 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Crossformer_weather_0.5_96_96_J_lambda1 \
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
  --top_k 5 \
  --des 'Exp' \
  --itr 1 
