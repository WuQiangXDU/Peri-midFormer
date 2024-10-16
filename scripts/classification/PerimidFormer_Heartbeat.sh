python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model PerimidFormer \
  --data UEA \
  --layers 2 \
  --batch_size 64 \
  --d_model 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0005 \
  --train_epochs 20 \
  --patience 5 \
  --chan_in 61