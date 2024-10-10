export CUDA_VISIBLE_DEVICES=0

bash ./scripts/Crossformer/ETTm1/Crossformer_R/Crossformer_R_no_imputate.sh
bash ./scripts/Crossformer/ETTm1/Crossformer_R/Crossformer_R_linear_imputate.sh
bash ./scripts/Crossformer/ETTm1/Crossformer_R/Crossformer_R_nearest_imputate.sh
bash ./scripts/Crossformer/ETTm1/Crossformer_I/Crossformer_I.sh