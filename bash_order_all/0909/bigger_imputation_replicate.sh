export CUDA_VISIBLE_DEVICES=0

#Informer
bash ./scripts/imputation/ETT_script/Informer_ETTh1_bigger_mr.sh
bash ./scripts/imputation/ETT_script/Informer_ETTm1_bigger_mr.sh
bash ./scripts/imputation/ECL_script/Informer_bigger_mr.sh
bash ./scripts/imputation/Weather_script/Informer_bigger_mr.sh

#PatchTST
bash ./scripts/imputation/ETT_script/PatchTST_ETTh1_bigger_mr.sh
bash ./scripts/imputation/ETT_script/PatchTST_ETTm1_bigger_mr.sh
bash ./scripts/imputation/ECL_script/PatchTST_bigger_mr.sh
bash ./scripts/imputation/Weather_script/PatchTST_bigger_mr.sh

#Crossformer
bash ./scripts/imputation/ETT_script/Crossformer_ETTh1_bigger_mr.sh
bash ./scripts/imputation/ETT_script/Crossformer_ETTm1_bigger_mr.sh
bash ./scripts/imputation/ECL_script/Crossformer_bigger_mr.sh
bash ./scripts/imputation/Weather_script/Crossformer_bigger_mr.sh

#iTransformer
bash ./scripts/imputation/ETT_script/iTransformer_ETTh1_bigger_mr.sh
bash ./scripts/imputation/ETT_script/iTransformer_ETTm1_bigger_mr.sh
bash ./scripts/imputation/ECL_script/iTransformer_bigger_mr.sh
bash ./scripts/imputation/Weather_script/iTransformer_bigger_mr.sh