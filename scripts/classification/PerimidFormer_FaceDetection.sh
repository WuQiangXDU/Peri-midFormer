python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model PerimidFormer \
  --data UEA \
  --layers 1 \
  --batch_size 2 \
  --d_model 32 \
  --top_k 2 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --chan_in 144