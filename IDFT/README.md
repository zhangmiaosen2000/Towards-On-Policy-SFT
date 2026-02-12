## 📦 Installation
Our codebase has been tested on H100 servers with the following environment:

* `python 3.12.0`
* `torch 2.8.0+cu128`

```bash
git clone https://github.com/zhangmiaosen2000/Towards-On-Policy-SFT
cd IDFT
```

### 🔧 Set Up Training Environment

```bash
conda create -n IDFT python=3.12 -y
conda activate IDFT
cd Towards-On-Policy-SFT/IDFT
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

The above only provides one way to configure the environment. For more detailed configuration instructions, please refer to the [verl documentation](https://verl.readthedocs.io/en/latest/start/install.html).


## 🚀 Getting Started

### Step 1: Prepare Data
You can generate data using the [Hinted Decoding](../HintedDecoding) approach. We have deployed this method in another subdirectory. The method provided here is specifically designed to generate data for the [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset:

```bash
python examples/data_preprocess/numina_cot.py \
  --local_dir PATH_TO_YOUR_DATA \
  --train_start 0 \
  --train_end 10000
```
If you also need to generate the test set, you can enable it by adding the `--create_test` argument.

### Step 2: Launch Training

```bash
set -x

nproc_per_node=8
project_name=numina-cot-origin-polylog

experiment_name=reexp-numina-cot-origin-polylog-qwen-2.5-7B-instruct
save_path=/mnt/vast/data/miaosen/lys/final_checkpoints/numina_cot_origin/$experiment_name

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_polylog_trainer \
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
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    model.polylog_beta=1.0 \
    model.polylog_gamma_min=0.1 \
    model.polylog_gamma_max=2.0 \
    model.polylog_warmup_ratio=0.0 \
    model.polylog_transition_ratio=0.0 \
    trainer.default_local_dir=$save_path \
    trainer.stats_dir=/mnt/home/t-miazhang/miaosen \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=500 \
    trainer.save_freq=2000 \
    trainer.total_epochs=3 \
    trainer.max_ckpt_to_keep=5 \
    trainer.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
```
