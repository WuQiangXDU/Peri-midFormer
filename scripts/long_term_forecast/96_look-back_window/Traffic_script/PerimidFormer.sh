model_name=PerimidFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --layers 1 \
  --chan_in 862 \
  --d_model 128 \
  --top_k 2 \
  --des 'Exp' \
  --itr 3 \
  --batch_size 2 \
  --learning_rate 0.005 \
  --train_epochs 15 \
  --patience 3


python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_96_192 \
 --model $model_name \
 --data custom \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 192 \
 --layers 1 \
 --chan_in 862 \
 --d_model 128 \
 --top_k 2 \
 --des 'Exp' \
 --itr 3 \
 --batch_size 2 \
 --learning_rate 0.005 \
 --train_epochs 15 \
 --patience 3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_96_336 \
 --model $model_name \
 --data custom \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 336 \
 --layers 1 \
 --chan_in 862 \
 --d_model 128 \
 --top_k 2 \
 --des 'Exp' \
 --itr 3 \
 --learning_rate 0.001 \
 --batch_size 2 \
 --train_epochs 15 \
 --patience 3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_96_720 \
 --model $model_name \
 --data custom \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 720 \
 --layers 1 \
 --chan_in 862 \
 --d_model 128 \
 --top_k 2 \
 --des 'Exp' \
 --itr 3 \
 --batch_size 2 \
 --learning_rate 0.001 \
 --train_epochs 15 \
 --patience 3