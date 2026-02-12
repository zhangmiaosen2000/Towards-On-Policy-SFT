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
You can generate data using the Hinted Decoding approach. We have deployed this method in another subdirectory. The method provided here is specifically designed to generate data for the [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset:

```bash
python examples/data_preprocess/numina_cot.py \
  --local_dir PATH_TO_YOUR_DATA \
  --train_start 0 \
  --train_end 10000
```
If you also need to generate the test set, you can enable it by adding the `--create_test` argument.
