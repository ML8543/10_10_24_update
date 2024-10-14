export CUDA_VISIBLE_DEVICES=0

#bash ./scripts/linear_nearest_only/Informer_ETTh1.sh
#bash ./scripts/linear_nearest_only/Informer_ECL.sh
#bash ./scripts/linear_nearest_only/Informer_ETTm1.sh
#bash ./scripts/linear_nearest_only/Informer_weather.sh
bash ./scripts/linear_nearest_only/PatchTST_ETTh1.sh
bash ./scripts/linear_nearest_only/PatchTST_ECL.sh
bash ./scripts/linear_nearest_only/PatchTST_ETTm1.sh
bash ./scripts/linear_nearest_only/PatchTST_weather.sh
bash ./scripts/linear_nearest_only/Crossformer_ETTh1.sh
bash ./scripts/linear_nearest_only/Crossformer_ECL.sh
bash ./scripts/linear_nearest_only/Crossformer_ETTm1.sh
bash ./scripts/linear_nearest_only/Crossformer_weather.sh
bash ./scripts/linear_nearest_only/iTransformer_ETTh1.sh
bash ./scripts/linear_nearest_only/iTransformer_ECL.sh
bash ./scripts/linear_nearest_only/iTransformer_ETTm1.sh
bash ./scripts/linear_nearest_only/iTransformer_weather.sh

shutdown