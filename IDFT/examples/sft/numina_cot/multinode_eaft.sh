set -x

nproc_per_node=8
project_name=numina80k-eaft

experiment_name=numina80k-eaft-qwen-2.5-7B-instruct
save_path=/mnt/vast/data/miaosen/lys/checkpoints/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    +data.multiturn.tools_key=tools \
    data.train_batch_size=128 \
    data.max_length=4096 \
    data.micro_batch_size_per_gpu=2 \
    +data.pad_mode=right \
    data.truncation=right \
    optim.lr=5e-6 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=0.1 \
    optim.lr_scheduler=cosine \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    trainer.total_epochs=3 \
    trainer.max_ckpt_to_keep=1 \
    trainer.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true \
    $@
