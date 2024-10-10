export CUDA_VISIBLE_DEVICES=0

#iTransformer_ETTh1
bash ./scripts/iTransformer/ETTh1/iTransformer_I/iTransformer_I.sh
bash ./scripts/iTransformer/ETTh1/iTransformer_R/Crossformer_R_no_imputate.sh
bash ./scripts/iTransformer/ETTh1/iTransformer_R/Crossformer_R_linear_imputate.sh
bash ./scripts/iTransformer/ETTh1/iTransformer_R/Crossformer_R_nearest_imputate.sh
bash ./scripts/iTransformer/ETTh1/iTransformer_J/lambda_not_stu/iTransformer_J_lambda0.sh
bash ./scripts/iTransformer/ETTh1/iTransformer_J/lambda_not_stu/iTransformer_J_lambda05.sh
bash ./scripts/iTransformer/ETTh1/iTransformer_J/lambda_not_stu/iTransformer_J_lambda1.sh
bash ./scripts/iTransformer/ETTh1/iTransformer_J/lambda_stu/iTransformer_J_ReLu.sh

#iTransformer_ETTm1
bash ./scripts/iTransformer/ETTm1/iTransformer_I/iTransformer_I.sh
bash ./scripts/iTransformer/ETTm1/iTransformer_R/Crossformer_R_no_imputate.sh
bash ./scripts/iTransformer/ETTm1/iTransformer_R/Crossformer_R_linear_imputate.sh
bash ./scripts/iTransformer/ETTm1/iTransformer_R/Crossformer_R_nearest_imputate.sh
bash ./scripts/iTransformer/ETTm1/iTransformer_J/lambda_not_stu/iTransformer_J_lambda0.sh
bash ./scripts/iTransformer/ETTm1/iTransformer_J/lambda_not_stu/iTransformer_J_lambda05.sh
bash ./scripts/iTransformer/ETTm1/iTransformer_J/lambda_not_stu/iTransformer_J_lambda1.sh
bash ./scripts/iTransformer/ETTm1/iTransformer_J/lambda_stu/iTransformer_J_ReLu.sh

#iTransformer_weather
bash ./scripts/iTransformer/weather/iTransformer_I/Informer_I.sh
bash ./scripts/iTransformer/weather/iTransformer_R/Informer_no_imputation.sh
bash ./scripts/iTransformer/weather/iTransformer_R/Informer_linear_imputation.sh
bash ./scripts/iTransformer/weather/iTransformer_R/Informer_nearest_imputation.sh
bash ./scripts/iTransformer/weather/iTransformer_J/lambda_not_stu/Informer_J_lambda0.sh
bash ./scripts/iTransformer/weather/iTransformer_J/lambda_not_stu/Informer_J_lambda05.sh
bash ./scripts/iTransformer/weather/iTransformer_J/lambda_not_stu/Informer_J_lambda1.sh
bash ./scripts/iTransformer/weather/iTransformer_J/lambda_stu/Informer_J.sh