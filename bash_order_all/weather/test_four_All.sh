export CUDA_VISIBLE_DEVICES=0

#imputation
#weather_Informer
bash ./scripts/imputation/Weather_script/Informer.sh
#weather_PatchTST
bash ./scripts/imputation/Weather_script/PatchTST.sh
#weather_Crossformer
bash ./scripts/imputation/Weather_script/Crossformer.sh
#itransformer
bash ./scripts/imputation/ETT_script/iTransformer_ETTm1.sh
bash ./scripts/imputation/Weather_script/iTransformer.sh

#long_term_forecast
#weather_Informer
bash ./scripts/long_term_forecast/Weather_script/Informer_96_96.sh
#weather_PatchTST
bash ./scripts/long_term_forecast/Weather_script/PatchTST_96_96.sh
#weather_Crossformer
bash ./scripts/long_term_forecast/Weather_script/Crossformer_96_96.sh
#itransformer
bash ./scripts/long_term_forecast/ETT_script/iTransformer_ETThm1_96_96.sh
bash ./scripts/long_term_forecast/Weather_script/iTransformer_96_96.sh