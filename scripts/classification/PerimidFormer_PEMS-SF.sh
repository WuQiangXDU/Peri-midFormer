python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path dataset/PEMS-SF/ \
  --model_id PEMS-SF \
  --model PerimidFormer \
  --data UEA \
  --layers 1 \
  --batch_size 16 \
  --d_model 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --chan_in 963