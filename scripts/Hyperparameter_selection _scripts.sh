# An example of the long-term forecasting task on the Electricity dataset

for d_model in 64 128 256 512 768
do
for layers in {1..5}
do
for top_k in {2..5}
do
for batch_size in 4 8 16 32 64
do
for learning_rate in 0.0001 0.0002 0.0005 0.001 0.002
do

python -u run.py \
--layers $layers \
--d_model $d_model \
--top_k $top_k \
--learning_rate $learning_rate \
--batch_size $batch_size \
--......

done
done
done
done
done
