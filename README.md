# KoBART Summarization Experiments

# Install Packages

bash install_kobart.sh
pip install -r requirements.txt

# Model Train
learning rate : 1e-5, 2e-5, 3e-5, 5e-5
bash ./run_train.sh

# Model Evaluation

1) Performance Change with Learning Rate (Beam Size 1)
CUDA_VISIBLE_DEVICES=0 python evaluate.py --checkpoint_path logs_lr1e-5 --beam_size 1
CUDA_VISIBLE_DEVICES=1 python evaluate.py --checkpoint_path logs_lr2e-5 --beam_size 1
CUDA_VISIBLE_DEVICES=2 python evaluate.py --checkpoint_path logs_lr3e-5 --beam_size 1
CUDA_VISIBLE_DEVICES=3 python evaluate.py --checkpoint_path logs_lr5e-5 --beam_size 1

2) Performance Change with Beam Size (learning rate : 3e-5 Model)

CUDA_VISIBLE_DEVICES=0 python evaluate.py --checkpoint_path logs_lr3e-5 --beam_size 1
CUDA_VISIBLE_DEVICES=1 python evaluate.py --checkpoint_path logs_lr3e-5 --beam_size 2
CUDA_VISIBLE_DEVICES=2 python evaluate.py --checkpoint_path logs_lr3e-5 --beam_size 4
CUDA_VISIBLE_DEVICES=3 python evaluate.py --checkpoint_path logs_lr3e-5 --beam_size 8
