python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path dataset/SWaT \
  --model_id SWAT \
  --model PerimidFormer \
  --data SWAT \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --layers 1 \
  --chan_in 51 \
  --top_k 2 \
  --anomaly_ratio 1 \
  --batch_size 8 \
  --train_epochs 1 \
  --itr 3 \
  --learning_rate 0.0001