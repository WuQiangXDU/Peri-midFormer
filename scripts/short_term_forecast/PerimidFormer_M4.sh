model_name=PerimidFormer

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --layers 2 \
  --chan_in 1 \
  --batch_size 1 \
  --d_model 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --layers 2 \
  --chan_in 1 \
  --batch_size 4 \
  --d_model 256 \
  --top_k 2 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0002 \
  --loss 'SMAPE' \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --layers 2 \
  --chan_in 1 \
  --batch_size 4 \
  --d_model 256 \
  --top_k 2 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0002 \
  --loss 'SMAPE' \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --layers 2 \
  --chan_in 1 \
  --batch_size 1 \
  --d_model 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --layers 2 \
  --chan_in 1 \
  --batch_size 1 \
  --d_model 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --layers 2 \
  --chan_in 1 \
  --batch_size 1 \
  --d_model 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --train_epochs 15 \
  --patience 3
