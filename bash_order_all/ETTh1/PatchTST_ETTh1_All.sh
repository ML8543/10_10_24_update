export CUDA_VISIBLE_DEVICES=0

#various imputation
bash ./scripts/PatchTST/ETTh1/Informer_R/PatchTST_R_no_imputate.sh
bash ./scripts/PatchTST/ETTh1/Informer_R/PatchTST_R_imputate_linear.sh
bash ./scripts/PatchTST/ETTh1/Informer_R/PatchTST_R_imputate_nearest.sh
#independent_trained_imputation
bash ./scripts/PatchTST/ETTh1/Informer_I/PatchTST_I.sh
#lambda_not_stu
bash ./scripts/PatchTST/ETTh1/Informer_J/lambda_not_stu/PatchTST_J_lambda0.sh
bash ./scripts/PatchTST/ETTh1/Informer_J/lambda_not_stu/PatchTST_J_lambda05.sh
bash ./scripts/PatchTST/ETTh1/Informer_J/lambda_not_stu/PatchTST_J_lambda1.sh
#lambda_stu
bash ./scripts/PatchTST/ETTh1/Informer_J/lambda_stu/PatchTST_J_ReLu.sh