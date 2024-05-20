model_name=PerimidFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_64_96 \
  --model $model_name \
  --data custom \
  --seq_len 64 \
  --label_len 48 \
  --pred_len 96 \
  --layers 5 \
  --chan_in 8 \
  --d_model 128 \
  --top_k 4 \
  --des 'Exp' \
  --itr 3 \
  --batch_size 4 \
  --learning_rate 0.005 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model_id Exchange_64_192 \
 --model $model_name \
 --data custom \
 --seq_len 64 \
 --label_len 48 \
 --pred_len 192 \
 --layers 3 \
 --chan_in 8 \
 --d_model 128 \
 --top_k 3 \
 --des 'Exp' \
 --itr 3 \
 --batch_size 16 \
 --learning_rate 0.005 \
 --train_epochs 15 \
 --patience 3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model_id Exchange_64_336 \
 --model $model_name \
 --data custom \
 --seq_len 64 \
 --label_len 48 \
 --pred_len 336 \
 --layers 3 \
 --chan_in 8 \
 --d_model 128 \
 --top_k 2 \
 --des 'Exp' \
 --itr 3 \
 --batch_size 8 \
 --learning_rate 0.005 \
 --train_epochs 15 \
 --patience 3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model_id Exchange_64_720 \
 --model $model_name \
 --data custom \
 --seq_len 64 \
 --label_len 48 \
 --pred_len 720 \
 --layers 3 \
 --chan_in 8 \
 --d_model 128 \
 --top_k 3 \
 --des 'Exp' \
 --itr 3 \
 --batch_size 8 \
 --learning_rate 0.005 \
 --train_epochs 15 \
 --patience 3