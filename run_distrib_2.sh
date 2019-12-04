export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

python -m torch.distributed.launch \
    --nproc_per_node=8 main.py \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 20 \
    --logging_steps 10 \
    --kl_model_type no \
    --use_discount \
    --use_random_position

# screen -S dialog_pretrain -L -Logfile pretrain.log