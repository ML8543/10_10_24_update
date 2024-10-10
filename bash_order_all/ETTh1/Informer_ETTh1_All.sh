export CUDA_VISIBLE_DEVICES=0

#various imputation
bash ./scripts/Informer/ETTh1/Informer_R/Informer_R_no_imputate.sh
bash ./scripts/Informer/ETTh1/Informer_R/Informer_R_linear.sh
bash ./scripts/Informer/ETTh1/Informer_R/Informer_R_nearest.sh
#independent_trained_imputation
bash ./scripts/Informer/ETTh1/Informer_I/Informer_I.sh
#lambda_not_stu
bash ./scripts/Informer/ETTh1/Informer_J/lambda_not_study/Informer_J_lambda0.sh
bash ./scripts/Informer/ETTh1/Informer_J/lambda_not_study/Informer_J_lambda05.sh
bash ./scripts/Informer/ETTh1/Informer_J/lambda_not_study/Informer_J_lambda1.sh
#lambda_stu
bash ./scripts/Informer/ETTh1/Informer_J/lambda_study/Informer_J_ReLu.sh
