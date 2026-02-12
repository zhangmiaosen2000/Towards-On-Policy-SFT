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
cd verl
bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```
