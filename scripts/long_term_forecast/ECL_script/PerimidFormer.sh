model_name=PerimidFormer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_96 \
  --model $model_name \
  --data custom \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --layers 2 \
  --chan_in 321 \
  --d_model 768 \
  --top_k 2 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --batch_size 4 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_192 \
  --model $model_name \
  --data custom \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --layers 2 \
  --chan_in 321 \
  --d_model 768 \
  --top_k 2 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --batch_size 4 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_336 \
  --model $model_name \
  --data custom \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --layers 2 \
  --chan_in 321 \
  --d_model 768 \
  --top_k 2 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --batch_size 4 \
  --patience 3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_720 \
  --model $model_name \
  --data custom \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --layers 2 \
  --chan_in 321 \
  --d_model 768 \
  --top_k 2 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --batch_size 4 \
  --patience 3