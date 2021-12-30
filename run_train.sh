CUDA_VISIBLE_DEVICES=0 python train.py \
--gradient_clip_val 1.0 \
--max_epochs 10 \
--default_root_dir logs_lr1e-5 \
--gpus 1 \
--batch_size 4 \
--num_workers 4 \
--lr 1e-5

CUDA_VISIBLE_DEVICES=1 python train.py \
--gradient_clip_val 1.0 \
--max_epochs 10 \
--default_root_dir logs_lr2e-5 \
--gpus 1 \
--batch_size 4 \
--num_workers 4 \
--lr 2e-5

CUDA_VISIBLE_DEVICES=2 python train.py \
--gradient_clip_val 1.0 \
--max_epochs 10 \
--default_root_dir logs_lr3e-5 \
--gpus 1 \
--batch_size 4 \
--num_workers 4 \
--lr 3e-5

CUDA_VISIBLE_DEVICES=3 python train.py \
--gradient_clip_val 1.0 \
--max_epochs 10 \
--default_root_dir logs_lr5e-5 \
--gpus 1 \
--batch_size 4 \
--num_workers 4 \
--lr 5e-5
