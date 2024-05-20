model_name=PerimidFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_96 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --layers 1 \
  --chan_in 7 \
  --d_model 16 \
  --top_k 3 \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_192 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --layers 2 \
  --chan_in 7 \
  --d_model 16 \
  --top_k 4 \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_336 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --layers 2 \
  --chan_in 7 \
  --d_model 16 \
  --top_k 4 \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_720 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --layers 2 \
  --chan_in 7 \
  --d_model 16 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3
