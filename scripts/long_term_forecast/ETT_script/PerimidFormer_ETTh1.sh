model_name=PerimidFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --layers 2 \
  --chan_in 7 \
  --d_model 512 \
  --des 'Exp' \
  --itr 3 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --top_k 5 \
  --train_epochs 15 \
  --patience 3


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_192 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --layers 1 \
  --chan_in 7 \
  --d_model 512 \
  --des 'Exp' \
  --itr 3 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --top_k 4 \
  --train_epochs 15 \
  --patience 3


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_336 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --layers 2 \
  --chan_in 7 \
  --d_model 512 \
  --des 'Exp' \
  --itr 3 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --top_k 2 \
  --train_epochs 15 \
  --patience 3


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_720 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --layers 2 \
  --chan_in 7 \
  --d_model 512 \
  --des 'Exp' \
  --itr 3 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --top_k 5 \
  --train_epochs 15 \
  --patience 3