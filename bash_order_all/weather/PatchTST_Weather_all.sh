export CUDA_VISIBLE_DEVICES=0

#I
bash ./scripts/PatchTST/weather/PatchTST_I/PatchTST_I.sh
#R
bash ./scripts/PatchTST/weather/PatchTST_R/PatchTST_R_no_imputate.sh
bash ./scripts/PatchTST/weather/PatchTST_R/PatchTST_R_linear_imputate.sh
bash ./scripts/PatchTST/weather/PatchTST_R/PatchTST_R_nearest_imputate.sh
#J
bash ./scripts/PatchTST/weather/PatchTST_J/lambda_not_stu/PatchTST_J_lambda0.sh
bash ./scripts/PatchTST/weather/PatchTST_J/lambda_not_stu/PatchTST_J_lambda05.sh
bash ./scripts/PatchTST/weather/PatchTST_J/lambda_not_stu/PatchTST_J_lambda1.sh
bash ./scripts/PatchTST/weather/PatchTST_J/lambda_stu/PatchTST_J_ReLu.sh