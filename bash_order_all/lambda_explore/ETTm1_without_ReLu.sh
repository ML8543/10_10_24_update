export CUDA_VISIBLE_DEVICES=0

#Informer
bash ./scripts/Informer/ETTm1/Informer_J/lambda_stu/Informer_J_ReLu.sh
bash ./scripts/Informer/ETTm1/Informer_J/lambda_stu/Informer_J_ReLu_bigger_mr.sh
#PatchTST
bash ./scripts/PatchTST/ETTm1/PatchTST_J/lambda_stu/PatchTST_J_ReLu.sh
bash ./scripts/PatchTST/ETTm1/PatchTST_J/lambda_stu/PatchTST_J_ReLu_bigger_mr.sh
#Crossformer
bash ./scripts/Crossformer/ETTm1/Crossformer_J/lambda_stu/Crossformer_J_ReLu.sh
bash ./scripts/Crossformer/ETTm1/Crossformer_J/lambda_stu/Crossformer_J_ReLu_bigger_mr.sh
#iTransformer
bash ./scripts/iTransformer/ETTm1/iTransformer_J/lambda_stu/iTransformer_J_ReLu.sh
bash ./scripts/iTransformer/ETTm1/iTransformer_J/lambda_stu/iTransformer_J_ReLu_bigger_mr.sh
#对lambda作出的修改就是把他的默认初始值由0改为0.5，并且修改exp中lambda的学习率为0.001