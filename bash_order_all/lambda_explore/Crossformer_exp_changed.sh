export CUDA_VISIBLE_DEVICES=0


#ETTh1
bash ./scripts/Crossformer/ETTh1/Crossformer_J/lambda_stu/Crossformer_J_ReLu.sh
bash ./scripts/Crossformer/ETTh1/Crossformer_J/lambda_stu/Crossformer_J_ReLu_bigger_mr.sh
#ETTm1
bash ./scripts/Crossformer/ETTm1/Crossformer_J/lambda_stu/Crossformer_J_ReLu.sh
bash ./scripts/Crossformer/ETTm1/Crossformer_J/lambda_stu/Crossformer_J_ReLu_bigger_mr.sh
#weather
bash ./scripts/Crossformer/weather/Crossformer_J/lambda_stu/Crossformer_J_ReLu.sh
bash ./scripts/Crossformer/weather/Crossformer_J/lambda_stu/Crossformer_J_ReLu_bigger_mr.sh
#对lambda作出的修改就是把他的默认初始值由0改为0.5，并且修改exp中lambda的学习率为0.001
shutdown