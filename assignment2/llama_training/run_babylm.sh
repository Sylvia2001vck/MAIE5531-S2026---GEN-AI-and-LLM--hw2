# Uncomment and set a real key only if you want wandb logging.
# export WANDB_API_KEY=your_real_wandb_api_key

python -u run_llama.py \
  --run_name run6-fix-loss \
  --option pretrain \
  --data_path train_100M \
  --block_size  128 \
  --batch_size 128 \
  --micro_batch_size 16 \
  --epochs 1 \
  --max_steps 300 \
  --log_every_steps 10 \
  --tokenized_dir train_100M/tokenized \
  --use_gpu  \
  --val_path dev \
  --val_tokenized_dir dev/tokenized \
  --val_per_steps 50 \
  --test_path  test \
  --test_tokenized_dir test/tokenized \
  --warmup_ratio 0.1 \
  --lr 1e-3 
# Add --auto_resume only when you intentionally want to continue a prior run.
# --overwrite_tokenized # if you want to overwrite the tokenized data