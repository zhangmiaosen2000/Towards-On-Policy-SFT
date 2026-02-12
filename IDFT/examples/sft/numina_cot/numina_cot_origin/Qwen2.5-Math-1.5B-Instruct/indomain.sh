set -x

nproc_per_node=8
project_name=numina-cot-origin-indomain

experiment_name=numina-cot-origin-idft-qwen-2.5-math-1.5B-instruct
save_path=/mnt/vast/data/miaosen/lys/final_checkpoints/numina_cot_origin/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_idft_trainer \
    data.train_files=/mnt/home/t-miazhang/miaosen/data/numina_333k/numina_333k.parquet \
    data.val_files=/mnt/vast/data/miaosen/lys/datasets/numina_cot/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=128 \
    data.max_length=4096 \
    data.micro_batch_size_per_gpu=2 \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    optim.lr=6e-5 \
    optim.lr_scheduler=cosine \
    model.partial_pretrain=Qwen/Qwen2.5-Math-1.5B-Instruct \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    model.indomain_tau=0.0 \
    model.indomain_temperature=1.0 \
    model.indomain_w_min=0.5 \
    model.indomain_w_max=1.5 \
    model.indomain_warmup_ratio=0.2 \
    model.indomain_transition_ratio=0.1 \
    trainer.default_local_dir=$save_path \
    trainer.stats_dir=/mnt/home/t-miazhang/miaosen \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    trainer.total_epochs=3 \
    trainer.max_ckpt_to_keep=5 \
    trainer.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true