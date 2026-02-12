set -x

nproc_per_node=8
project_name=numina-cot

experiment_name=numina-cot-polylog-qwen-2.5-7B
save_path=/mnt/vast/data/miaosen/lys/checkpoints/$experiment_name

# Adaptive Poly-Log method
# Loss: L(p) = -(p^γ) * log(p)
# γ = exp(-φ/β), where φ = log p + H(p)
# Out-domain (φ<<0) → γ large → SFT behavior
# In-domain (φ≈0) → γ≈1 → DFT behavior  
# Strong (φ>0) → γ small → Ignore behavior

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_polylog_trainer \
    data.train_files=/mnt/vast/data/miaosen/lys/datasets/numina_cot/train.parquet \
    data.val_files=/mnt/vast/data/miaosen/lys/datasets/numina_cot/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=128 \
    data.max_length=16384 \
    data.micro_batch_size_per_gpu=1 \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    optim.lr=5e-6 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=0.1 \
    optim.lr_scheduler=cosine \
    model.partial_pretrain=Qwen/Qwen2.5-7B \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    model.polylog_beta=1.0 \
    model.polylog_gamma_min=0.1 \
    model.polylog_gamma_max=2.0 \
    model.polylog_warmup_ratio=0.2 \
    model.polylog_transition_ratio=0.1 \
    trainer.default_local_dir=$save_path \
    trainer.stats_dir=/mnt/home/t-miazhang/miaosen \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=5 \
    trainer.save_freq=100 \
    trainer.total_epochs=3 \
    trainer.max_ckpt_to_keep=1 \
    trainer.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true
