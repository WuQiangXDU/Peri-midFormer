python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path dataset/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model PerimidFormer \
  --data UEA \
  --layers 1 \
  --batch_size 32 \
  --d_model 32 \
  --top_k 6 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --patience 5 \
  --chan_in 7