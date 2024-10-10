export CUDA_VISIBLE_DEVICES=0

model_name=Informer

# 外层循环
for learning_rate in 0.001 0.005 0.01 0.05
do
  # 中层循环
  for d_model in 16 32 64 128 256 512
  do
    # 内层循环
    for d_ff in 16 32 64 128 256 512
    do
      python -u run.py \
        --task_name imputation \
        --is_training 1 \
        --root_path ./dataset/illness/ \
        --data_path national_illness.csv \
        --model_id ILI_mask_0.25 \
        --mask_rate 0.25 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 0 \
        --pred_len 0 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --batch_size 16 \
        --d_model $d_model \
        --d_ff $d_ff \
        --des 'Exp' \
        --itr 1 \
        --top_k 5 \
        --learning_rate $learning_rate
    done
  done
done