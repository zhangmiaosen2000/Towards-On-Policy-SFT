<div align="left">
  <a href="https://www.seu.edu.cn/" target="_blank" rel="noopener"><img src="https://raw.githubusercontent.com/seumxc/SEU-Logo/master/%E4%B8%9C%E5%8D%97%E5%A4%A7%E5%AD%A6%E6%A0%A1%E5%BE%BD/%E5%8E%9F%E9%85%8D%E8%89%B2.svg" alt="SEU" height="40" style="margin-right:10px; vertical-align:middle;" /></a>
  <a href=“https://www.microsoft.com/en-us/research/lab/microsoft-research-asia-zh-cn/" target="_blank" rel="noopener"><img src="https://upload.wikimedia.org/wikipedia/commons/4/44/Microsoft_logo.svg" alt="Microsoft" height="38" style="margin-right:10px; vertical-align:middle;" /></a>
  <a href="https://shopee.com/" target="_blank" rel="noopener"><img src="https://upload.wikimedia.org/wikipedia/commons/0/0e/Shopee_logo.svg" alt="Shopee" height="40" style="vertical-align:middle;" /></a>
</div>



<div align="center">
  
# *Towards On-Policy SFT*: <br>Distribution Discriminant Theory and <br>its Applications in LLM Training

<a href="javascript:void(0)" target="_blank">
  <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Pending-red?logo=arxiv" height="25" />
</a>

<div align="center">

<p>
  <b>Authors</b><br/>
  <a href="#" target="_blank">Miaosen Zhang</a><sup>1,2</sup>*
  · <a href="#" target="_blank">Yishan Liu</a><sup>1,3</sup>*
  · <a href="#" target="_blank">Shuxia Lin</a><sup>1</sup>
  · <a href="#" target="_blank">Xu Yang</a><sup>1</sup>
  · <a href="#" target="_blank">Qi Dai</a><sup>2</sup>
  · <a href="#" target="_blank">Chong Luo</a><sup>2</sup>
  <br/>
  <a href="#" target="_blank">Weihao Jiang</a><sup>3</sup>
  · <a href="#" target="_blank">Peng Hou</a><sup>3</sup>
  · <a href="#" target="_blank">Anxiang Zeng</a><sup>3</sup>
  · <a href="#" target="_blank">Xin Geng</a><sup>1</sup>
  · <a href="#" target="_blank">Baining Guo</a><sup>1,2</sup>
</p>

<!-- Affiliations -->
<p>
  <b>Affiliations</b><br/>
  <sup>1</sup> Southeast University (SEU) &nbsp;|&nbsp;
  <sup>2</sup> Microsoft Research Asia (MSRA) &nbsp;|&nbsp;
  <sup>3</sup> Shopee
</p>

</div>
</div>


## ⭐️ News

* **\[2026.02.13]** We have uploaded our paper to arXiv.

## 📄 Abstract
Supervised fine-tuning (SFT) is computationally efficient but often yields inferior generalization compared to reinforcement learning (RL). This gap is primarily driven by RL’s use of on-policy data. We propose a framework to bridge this chasm by enabling On-Policy SFT. We first present Distribution Discriminant Theory (DDT), which explains and quantifies the alignment between data and the model-induced distribution. Leveraging DDT, we introduce two complementary techniques: (i) In-Distribution Finetuning (IDFT), a loss-level method to enhance generalization ability of SFT, and (ii) Hinted Decoding, a data-level technique that can re-align the training corpus to the model’s distribution. Extensive experiments demonstrate that our framework achieves generalization performance on par with prominent offline RL algorithms, including DPO and SimPO, while maintaining the efficiency of an SFT pipeline. The proposed framework thus offers a practical alternative in domains where RL is infeasible.

## 🗓️ Release Plan
| Release Item | Status |
|---|---|
| Paper (arXiv) | [✅ Released]() |
| In-Domain Finetuning Training code | [✅ Released]() |
| Hinted Decoding Demo | [✅ Released](HintedDecoding/README.md) |
| Hinted Decoding Demo (vllm version) | ⏳ In progress |
| Re-conduct the exps in paper with better dataset (e.g., DeepMath), and more models to enhance the reliability | ⏳ TODO |
| The online version of the on-policy sft training (model updates and trains for each batch) | ⏳ TODO |
| Paper (arXiv) V2 Version  | ⏳ TODO |


We are doing more exps and efforts to make the technique more solid, please stay-turned!


## 📊 Experiment Results


## Contect Us

We are still continuously updating. We plan to add more experiments to enhance the impact of the study and update the version of the paper. In the meantime, we welcome your suggestions in all aspects. (t-miazhang@microsoft.com, yishan@xxx.com)