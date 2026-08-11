python -u run.py \
	--random_seed 2025 \
    --is_training 1 \
    --alpha 158 \
    --instruments csi300 \
    --model LSTM \
    --seq_len 30 \
    --horizon 5 \
    --train_begin_date 2022-01-01 \
    --valid_begin_date 2025-01-01 \
    --test_begin_date 2025-01-01 \
    --test_end_date 2025-08-31 \
    --train_epochs 10 \
    --batch_size 1 \
    --checkpoint ~/suyanning_nfs_data/quant/Learning-Feamework/checkpoint/LSTM \
    --result ~/suyanning_nfs_data/quant/Learning-Feamework/result/LSTM \
    --log ~/suyanning_nfs_data/quant/Learning-Feamework/logs/LSTM

cd ../../Learning-Feamework
cd result
python pkl2csv.py

cd ../backtest
python Test.py