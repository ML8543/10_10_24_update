export CUDA_VISIBLE_DEVICES=0

model_name=Informer

# 以下是修改后的命令行调用
python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5 \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration_classify_joint \
  --model "$model_name" \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

# 重复上面的模式，为其他数据集调用脚本
# ...