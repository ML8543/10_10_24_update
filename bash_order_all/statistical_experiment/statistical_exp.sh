export CUDA_VISIBLE_DEVICES=0

#Informer
bash scripts/statistical_experiment/ETTh1_Informer.sh
bash scripts/statistical_experiment/ECl_Informer.sh
bash scripts/statistical_experiment/ETTm1_Informer.sh
bash scripts/statistical_experiment/weather_Informer.sh
#PatchTST
bash scripts/statistical_experiment/ETTh1_PatchTST.sh
bash scripts/statistical_experiment/ECL_PatchTST.sh
bash scripts/statistical_experiment/ETTm1_PatchTST.sh
bash scripts/statistical_experiment/weather_PatchTST.sh
#Crossformer
bash scripts/statistical_experiment/ETTh1_Crossformer.sh
bash scripts/statistical_experiment/ECL_Crossformer.sh
bash scripts/statistical_experiment/ETTm1_Crossformer.sh
bash scripts/statistical_experiment/weather_Crossformer.sh
#iTransformer
bash scripts/statistical_experiment/ETTh1_iTransformer.sh
bash scripts/statistical_experiment/ECL_iTransformer.sh
bash scripts/statistical_experiment/ETTm1_iTransformer.sh
bash scripts/statistical_experiment/weather_iTransformer.sh
