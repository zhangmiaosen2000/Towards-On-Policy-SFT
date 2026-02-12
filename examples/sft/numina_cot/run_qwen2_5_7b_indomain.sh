set -x

nproc_per_node=8
project_name=numina-cot

experiment_name=numina-cot-indomain-qwen-2.5-math-1.5B-v4
save_path=/mnt/vast/data/miaosen/lys/checkpoints/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_indomain_trainer \
    data.train_files=/mnt/vast/data/miaosen/lys/datasets/numina_cot/train.parquet \
    data.val_files=/mnt/vast/data/miaosen/lys/datasets/numina_cot/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=256 \
    data.max_length=3800 \
    optim.lr=5e-5 \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=Qwen/Qwen2.5-Math-1.5B \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.indomain_tau=0.0 \
    model.indomain_temperature=0.7 \
    model.indomain_w_min=1 \
    model.indomain_w_max=2 \
    trainer.default_local_dir=$save_path \
    trainer.stats_dir=/mnt/home/t-miazhang/miaosen/qwen2_5_math_1.5b \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    trainer.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true