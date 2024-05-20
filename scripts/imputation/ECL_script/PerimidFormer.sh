model_name=PerimidFormer

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --layers 1 \
  --chan_in 321 \
  --batch_size 2 \
  --d_model 768 \
  --des 'Exp' \
  --itr 3 \
  --top_k 2 \
  --learning_rate 0.0005 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --layers 1 \
  --chan_in 321 \
  --batch_size 2 \
  --d_model 768 \
  --des 'Exp' \
  --itr 3 \
  --top_k 2 \
  --learning_rate 0.0005 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --layers 1 \
  --chan_in 321 \
  --batch_size 2 \
  --d_model 768 \
  --des 'Exp' \
  --itr 3 \
  --top_k 2 \
  --learning_rate 0.0005 \
  --train_epochs 15 \
  --patience 3

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --layers 1 \
  --chan_in 321 \
  --batch_size 2 \
  --d_model 768 \
  --des 'Exp' \
  --itr 3 \
  --top_k 2 \
  --learning_rate 0.0005 \
  --train_epochs 15 \
  --patience 3

