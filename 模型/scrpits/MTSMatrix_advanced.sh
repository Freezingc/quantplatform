python -u run.py \
	--random_seed 2025 \
    --is_training 0 \
    --inference 1 \
    --infer_date 2025-12-25 \
    --fast_inference 1 \
    --alpha 158 \
    --instruments all \
    --model MTSMatrix \
    --seq_len 30 \
    --horizon 5 \
    --enc_in 158 \
    --pred_len 1 \
    --e_layers 2 \
    --sampling 2 \
    --mat 0 \
    --norm 1 \
    --rev 1 \
    --train_begin_date 2022-01-01 \
    --valid_begin_date 2025-01-01 \
    --test_begin_date 2025-01-01 \
    --test_end_date 2025-08-29 \
    --train_epochs 250 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --use_gpu True \
    --gpu 2 \
    --patience 50 \
    --checkpoint /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/checkpoint/MTSMatrix \
    --result /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/result/MTSMatrix \
    --log /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/logs/MTSMatrix \



