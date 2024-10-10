export CUDA_VISIBLE_DEVICES=0

#I
bash ./scripts/Informer/waether/Informer_I/Informer_I.sh
#R
bash ./scripts/Informer/waether/Informer_R/Informer_no_imputation.sh
bash ./scripts/Informer/waether/Informer_R/Informer_linear_imputation.sh
bash ./scripts/Informer/waether/Informer_R/Informer_nearest_imputation.sh
#J
bash ./scripts/Informer/waether/Informer_J/lambda_not_stu/Informer_J_lambda0.sh
bash ./scripts/Informer/waether/Informer_J/lambda_not_stu/Informer_J_lambda05.sh
bash ./scripts/Informer/waether/Informer_J/lambda_not_stu/Informer_J_lambda1.sh
bash ./scripts/Informer/waether/Informer_J/lambda_stu/Informer_J.sh