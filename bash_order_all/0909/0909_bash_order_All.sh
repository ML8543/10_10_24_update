export CUDA_VISIBLE_DEVICES=0

#iTransformer_ECL
#I
bash ./scripts/iTransformer/ECL/iTransformer_I/iTransformer.sh
#R
bash ./scripts/iTransformer/ECL/iTransformer_R/Informer_R_no_imputate.sh
bash ./scripts/iTransformer/ECL/iTransformer_R/Informer_R_linear_imputate.sh
bash ./scripts/iTransformer/ECL/iTransformer_R/Informer_R_nearest_imputate.sh
#J
bash ./scripts/iTransformer/ECL/iTransformer_J/lambda_not_stu/Informer_J_lambda0.sh
bash ./scripts/iTransformer/ECL/iTransformer_J/lambda_not_stu/Informer_J_lambda05.sh
bash ./scripts/iTransformer/ECL/iTransformer_J/lambda_not_stu/Informer_J_lambda1.sh
bash ./scripts/iTransformer/ECL/iTransformer_J/lambda_stu/Informer_J_ReLu.sh

#Informer_ETTh1_bigger_rate
bash ./scripts/Informer/ETTh1/Informer_I/Informer_I_bigger_mr.sh
bash ./scripts/Informer/ETTh1/Informer_J/lambda_not_study/Informer_J_lambda_not_stu_bigger_mr.sh
bash ./scripts/Informer/ETTh1/Informer_J/lambda_study/Informer_J_ReLu_bigger_mr.sh
bash ./scripts/Informer/ETTh1/Informer_R/Informer_R_various_imputation_bigger_mr.sh

#Informer_ETTm1_bigger_rate
bash ./scripts/Informer/ETTm1/Informer_I/Informer_I_bigger_mr.sh
bash ./scripts/Informer/ETTm1/Informer_J/lambda_not_stu/Informer_J_lambda_not_stu_bigger_mr.sh
bash ./scripts/Informer/ETTm1/Informer_J/lambda_stu/Informer_J_ReLu_bigger_mr.sh
bash ./scripts/Informer/ETTm1/Informer_R/Informer_R_various_imputate_bigger_mr.sh

#Informer_ECL_bigger_rate
bash ./scripts/Informer/ECL/Informer_I/Informer_I_bigger_mr.sh
bash ./scripts/Informer/ECL/Informer_J/lambda_not_study/Informer_J_lambda_not_stu_bigger_mr.sh
bash ./scripts/Informer/ECL/Informer_J/lambda_study/Informer_J_ReLu_bigger_mr.sh
bash ./scripts/Informer/ECL/Informer_R/Informer_R_various_imputate_bigger_mr.sh

#Informer_weather_bigger_rate
bash ./scripts/Informer/waether/Informer_I/Informer_I_bigger_mr.sh
bash ./scripts/Informer/waether/Informer_J/lambda_not_stu/Informer_J_lambda_not_stu_bigger_mr.sh
bash ./scripts/Informer/waether/Informer_J/lambda_stu/Informer_J_bigger_mr.sh
bash ./scripts/Informer/waether/Informer_R/Informer_various_imputation_bigger_mr.sh

#PatchTST_ETTh1_bigger_rate
bash ./scripts/PatchTST/ETTh1/Informer_I/PatchTST_I_bigger_mr.sh
bash ./scripts/PatchTST/ETTh1/Informer_J/lambda_not_stu/PatchTST_J_lambda_not_stu_bigger_mr.sh
bash ./scripts/PatchTST/ETTh1/Informer_J/lambda_stu/PatchTST_J_ReLu_bigger_mr.sh
bash ./scripts/PatchTST/ETTh1/Informer_R/PatchTST_R_various_imputate_bigger_mr.sh

#PatchTST_ETTm1_bigger_rate
bash ./scripts/PatchTST/ETTm1/PatchTST_I/PatchTST_I_bigger_mr.sh
bash ./scripts/PatchTST/ETTm1/PatchTST_J/lambda_not_stu/PatchTST_J_lambda_not_stu_bigger_mr.sh
bash ./scripts/PatchTST/ETTm1/PatchTST_J/lambda_stu/PatchTST_J_ReLu_bigger_mr.sh
bash ./scripts/PatchTST/ETTm1/PatchTST_R/PatchTST_R_various_imputate_bigger_mr.sh

#PatchTST_ECL_bigger_rate
bash ./scripts/PatchTST/ECL/Informer_I/PatchTST_I_bigger_mr.sh
bash ./scripts/PatchTST/ECL/Informer_J/lambda_not_study/PatchTST_J_lambda_not_stu_bigger_mr.sh
bash ./scripts/PatchTST/ECL/Informer_J/lambda_study/PatchTST_J_ReLu_bigger_mr.sh
bash ./scripts/PatchTST/ECL/Informer_R/PatchTST_R_various_imputate_bigger_mr.sh

#PatchTST_weather_bigger_rate
bash ./scripts/PatchTST/weather/PatchTST_I/PatchTST_I_bigger_mr.sh
bash ./scripts/PatchTST/weather/PatchTST_J/lambda_not_stu/PatchTST_J_lambda_bigger_mr.sh
bash ./scripts/PatchTST/weather/PatchTST_J/lambda_stu/PatchTST_J_ReLu_bigger_mr.sh
bash ./scripts/PatchTST/weather/PatchTST_R/PatchTST_R_various_imputate_bigger_mr.sh

#Crossformer_ETTh1_bigger_rate
bash ./scripts/Crossformer/ETTh1/Crossformer_I/Informer_I_bigger_mr.sh
bash ./scripts/Crossformer/ETTh1/Crossformer_J/lambda_not_stu/Crossformer_J_lambda_bigger_mr.sh
bash ./scripts/Crossformer/ETTh1/Crossformer_J/lambda_stu/Crossformer_J_ReLu_bigger_mr.sh
bash ./scripts/Crossformer/ETTh1/Crossformer_R/Crossformer_R_various_imputate_bigger_mr.sh

#Crossformer_ETTm1_bigger_rate
bash ./scripts/Crossformer/ETTm1/Crossformer_I/Crossformer_I_bigger_mr.sh
bash ./scripts/Crossformer/ETTm1/Crossformer_J/lambda_not_stu/Crossformer_J_lambda_bigger_mr.sh
bash ./scripts/Crossformer/ETTm1/Crossformer_J/lambda_stu/Crossformer_J_ReLu_bigger_mr.sh
bash ./scripts/Crossformer/ETTm1/Crossformer_R/Crossformer_R_various_imputate_bigger_mr.sh

#Crossformer_ECL_bigger_rate
bash ./scripts/Crossformer/ECL/Crossformer_I/Crossformer_I_bigger_mr.sh
bash ./scripts/Crossformer/ECL/Crossformer_J/lambda_not_stu/Crossformer_J_lambda_bigger_mr.sh
bash ./scripts/Crossformer/ECL/Crossformer_J/lambda_stu/Crossformer_J_ReLu_bigger_mr.sh
bash ./scripts/Crossformer/ECL/Crossformer_R/Crossformer_R_various_imputate_bigger_mr.sh

#Crossformer_weather_bigger_rate
bash ./scripts/Crossformer/weather/Crossformer_I/Informer_I_bigger_mr.sh
bash ./scripts/Crossformer/weather/Crossformer_J/lambda_not_stu/Crossformer_J_lambda_bigger_mr.sh
bash ./scripts/Crossformer/weather/Crossformer_J/lambda_stu/Crossformer_J_ReLu_bigger_mr.sh
bash ./scripts/Crossformer/weather/Crossformer_R/Crossformer_R_various_imputate_bigger_mr.sh

#iTransformer_ETTh1_bigger_rate
bash .scripts/iTransformer/ETTh1/iTransformer_I/iTransformer_I_bigger_mr.sh
bash .scripts/iTransformer/ETTh1/iTransformer_J/lambda_not_stu/iTransformer_J_lambda_bigger_mr.sh
bash .scripts/iTransformer/ETTh1/iTransformer_J/lambda_stu/iTransformer_J_ReLu_bigger_mr.sh
bash .scripts/iTransformer/ETTh1/iTransformer_R/Crossformer_R_no_imputate_bigger_mr.sh

#iTransformer_ETTm1_bigger_rate
bash .scripts/iTransformer/ETTm1/iTransformer_I/iTransformer_I_bigger_mr.sh
bash .scripts/iTransformer/ETTm1/iTransformer_J/lambda_not_stu/iTransformer_J_lambda0_bigger_mr.sh
bash .scripts/iTransformer/ETTm1/iTransformer_J/lambda_stu/iTransformer_J_ReLu_bigger_mr.sh
bash .scripts/iTransformer/ETTm1/iTransformer_R/Crossformer_R_various_imputate_bigger_mr.sh

#iTransformer_ECL_bigger_rate
bash .scripts/iTransformer/ECL/iTransformer_I/iTransformer_bigger_mr.sh
bash .scripts/iTransformer/ECL/iTransformer_J/lambda_not_stu/Informer_J_lambda_bigger_mr.sh
bash .scripts/iTransformer/ECL/iTransformer_J/lambda_stu/Informer_J_ReLu_bigger_mr.sh
bash .scripts/iTransformer/ECL/iTransformer_R/Informer_R_various_imputate_bigger_mr.sh

#iTransformer_weather_bigger_rate
bash .scripts/iTransformer/weather/iTransformer_I/Informer_I_bigger_mr.sh
bash .scripts/iTransformer/weather/iTransformer_J/lambda_not_stu/Informer_J_lambda0_bigger_mr.sh
bash .scripts/iTransformer/weather/iTransformer_J/lambda_stu/Informer_J_bigger_mr.sh
bash .scripts/iTransformer/weather/iTransformer_R/Informer_no_imputation_bigger_mr.sh