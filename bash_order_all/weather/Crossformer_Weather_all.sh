export CUDA_VISIBLE_DEVICES=0

#I
bash ./scripts/Crossformer/weather/Crossformer_I/Informer_I.sh
#R
bash ./scripts/Crossformer/weather/Crossformer_R/Crossformer_R_no_imputate.sh
bash ./scripts/Crossformer/weather/Crossformer_R/Crossformer_R_linear_imputate.sh
bash ./scripts/Crossformer/weather/Crossformer_R/Crossformer_R_nearest_imputate.sh
#J
bash ./scripts/Crossformer/weather/Crossformer_J/lambda_not_stu/Crossformer_J_lambda0.sh
bash ./scripts/Crossformer/weather/Crossformer_J/lambda_not_stu/Crossformer_J_lambda05.sh
bash ./scripts/Crossformer/weather/Crossformer_J/lambda_not_stu/Crossformer_J_lambda1.sh
bash ./scripts/Crossformer/weather/Crossformer_J/lambda_stu/Crossformer_J_ReLu.sh
