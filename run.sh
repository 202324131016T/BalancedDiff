# training
# CUDA_VISIBLE_DEVICES=xxx
# --model_idx xxx \   -> auto-complete:from models.our_model_xx import ScorePosNet3D
# --model_type xxx \  0:no loss balance   1:use loss balance
CUDA_VISIBLE_DEVICES=1 \
python scripts/our_train_diffusion.py \
--info_path ./data/info.pkl \
--config ./configs/our_training.yml \
--diffusion_logdir ./logs_diffusion/ \
--data_path ./data/ \
--model_type 1 \
--model_idx main


# Sampling for pockets in the testset  --data_id 0-99
# CUDA_VISIBLE_DEVICES=xxx
# --model_idx xxx \
# --diffusion_path ./logs_diffusion/xxx/checkpoints/best.pt
CUDA_VISIBLE_DEVICES=1 \
python scripts/our_sample_diffusion.py \
--config ./configs/our_sampling.yml \
--sample_logdir ./logs_sample/ \
--data_path ./data/ \
--data_id 0-99 \
--model_idx main \
--diffusion_path ./logs_diffusion/xxx/checkpoints/best.pt


# Evaluation: Evaluation from sampling results use CPU
# --eval_step 0 \ default -1
# --sample_path ./logs_sample/xxx/
python scripts/our_evaluate_diffusion.py \
--evaluate_logdir ./logs_evaluate/ \
--data_path ./data/test_set/ \
--eval_step -1 \
--sample_path ./logs_sample/xxx/


exit 0


# raw for the method TargetDiff
# raw training
CUDA_VISIBLE_DEVICES=0 \
python scripts/raw_train_diffusion.py \
--config ./configs/raw_training.yml \
--diffusion_logdir ./logs_diffusion/ \
--data_path ./data/

# raw sample
CUDA_VISIBLE_DEVICES=0 \
python scripts/raw_sample_diffusion.py \
 --config ./configs/raw_sampling.yml \
--data_id 0-99 \
--sample_logdir ./logs_sample/ \
--data_path ./data/ \
--diffusion_path ./logs_diffusion/xxx/checkpoints/10000.pt

# raw evaluate
python scripts/raw_evaluate_diffusion.py \
--evaluate_logdir ./logs_evaluate/ \
--data_path ./data/test_set/ \
--sample_path ./logs_sample/xxx/

