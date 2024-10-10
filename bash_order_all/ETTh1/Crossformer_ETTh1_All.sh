export CUDA_VISIBLE_DEVICES=0

#various imputation
bash ./scripts/Crossformer/ETTh1/Crossformer_R/Crossformer_R_no_imputate.sh
bash ./scripts/Crossformer/ETTh1/Crossformer_R/Crossformer_R_linear_imputate.sh
bash ./scripts/Crossformer/ETTh1/Crossformer_R/Crossformer_R_nearest_imputate.sh
#independent_trained_imputation
bash ./scripts/Crossformer/ETTh1/Crossformer_I/Informer_I.sh
#lambda_not_stu
bash ./scripts/Crossformer/ETTh1/Crossformer_J/lambda_not_stu/Crossformer_J_lambda0.sh
bash ./scripts/Crossformer/ETTh1/Crossformer_J/lambda_not_stu/Crossformer_J_lambda05.sh
bash ./scripts/Crossformer/ETTh1/Crossformer_J/lambda_not_stu/Crossformer_J_lambda1.sh
#lambda_stu
bash ./scripts/Crossformer/ETTh1/Crossformer_J/lambda_stu/Crossformer_J_ReLu.sh
