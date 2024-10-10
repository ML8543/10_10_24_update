export CUDA_VISIBLE_DEVICES=0

model_name=Informer

#############
python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5 \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/Handwriting/ \
  --model_id Handwriting_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/PEMS-SF/ \
  --model_id PEMS-SF_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits_classify_joint \
  --model $model_name \
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

python -u run_classify_jointly.py \
  --task_name classify_jointly \
  --mask_rate 0.125 \
  --_lambda 0.5
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary_classify_joint \
  --model $model_name \
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
