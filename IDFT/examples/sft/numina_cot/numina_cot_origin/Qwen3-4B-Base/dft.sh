set -x

nproc_per_node=8
project_name=numina-cot-origin-dft

experiment_name=numina-cot-origin-dft-qwen-3-4B-base
save_path=/mnt/vast/data/miaosen/lys/final_checkpoints/numina_cot_origin/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_dft_trainer \
    data.train_files=/mnt/vast/data/miaosen/lys/datasets/numina_cot/train.parquet \
    data.val_files=/mnt/vast/data/miaosen/lys/datasets/numina_cot/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=256 \
    data.max_length=3688 \
    data.micro_batch_size_per_gpu=4 \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    optim.lr=6e-5 \
    optim.lr_scheduler=cosine \
    model.partial_pretrain=Qwen/Qwen3-4B-Base \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    trainer.default_local_dir=$save_path \
    trainer.stats_dir=/mnt/home/t-miazhang/miaosen \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    trainer.total_epochs=1 \
    trainer.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true