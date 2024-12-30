model_name=PerimidFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --layers 1 \
  --chan_in 7 \
  --des 'Exp' \
  --d_model 512 \
  --top_k 2 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --layers 1 \
  --chan_in 7 \
  --des 'Exp' \
  --d_model 256 \
  --top_k 2 \
  --batch_size 2 \
  --learning_rate 0.0005 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --layers 1 \
  --chan_in 7 \
  --des 'Exp' \
  --d_model 256 \
  --top_k 2 \
  --batch_size 2 \
  --learning_rate 0.0005 \
  --itr 3 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --layers 1 \
  --chan_in 7 \
  --des 'Exp' \
  --d_model 128 \
  --top_k 2 \
  --itr 3 \
  --batch_size 2 \
  --learning_rate 0.005 \
  --train_epochs 15 \
  --patience 3