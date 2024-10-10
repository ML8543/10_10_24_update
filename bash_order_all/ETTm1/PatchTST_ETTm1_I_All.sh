export CUDA_VISIBLE_DEVICES=0

bash ./scripts/PatchTST/ETTm1/PatchTST_R/PatchTST_R_no_imputate.sh
bash ./scripts/PatchTST/ETTm1/PatchTST_R/PatchTST_R_linear_imputate.sh
bash ./scripts/PatchTST/ETTm1/PatchTST_R/PatchTST_R_nearest_imputate.sh
bash ./scripts/PatchTST/ETTm1/PatchTST_I/PatchTST_I.sh