model_name=PerimidFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --layers 1 \
  --chan_in 7 \
  --d_model 256 \
  --top_k 2 \
  --des 'Exp' \
  --batch_size 2 \
  --learning_rate 0.0005 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --layers 1 \
  --chan_in 7 \
  --d_model 128 \
  --top_k 2 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.005 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --layers 1 \
  --chan_in 7 \
  --d_model 256 \
  --top_k 4 \
  --des 'Exp' \
  --batch_size 2 \
  --learning_rate 0.001 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --layers 1 \
  --chan_in 7 \
  --d_model 256 \
  --top_k 2 \
  --des 'Exp' \
  --batch_size 4 \
  --learning_rate 0.005 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3
