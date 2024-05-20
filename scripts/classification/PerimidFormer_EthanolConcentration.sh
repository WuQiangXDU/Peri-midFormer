python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model PerimidFormer \
  --data UEA \
  --layers 4 \
  --batch_size 32 \
  --d_model 256 \
  --top_k 8 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  --chan_in 3