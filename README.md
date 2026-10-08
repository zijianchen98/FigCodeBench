<div align="center">

<div>
<a href="https://github.com/zijianchen98/FigCodeBench"><img src="https://visitor-badge.laobi.icu/badge?page_id=zijianchen98/FigCodeBench"/></a>
    <a href="https://github.com/zijianchen98/FigCodeBench"><img src="https://img.shields.io/github/stars/zijianchen98/FigCodeBench"/></a>
    <a href="https://arxiv.org/abs/2610.10066"><img src="https://img.shields.io/badge/Arxiv-2610:10066-red"/></a>
    <a href="https://github.com/zijianchen98/FigCodeBench"><img src="https://img.shields.io/badge/Dataset-Release-green"></a>
    <a href="https://github.com/zijianchen98/FigCodeBench"><img src="https://img.shields.io/badge/Awesome-FigCodeBench-orange"/></a>
</div>

<h1>From Pixel to Coding: Evaluating the Figure Reproduction Capabilities of MLLMs</h1>

_Evaluating MLLMs on figure reproduction, integrating multimodal comprehension and generation._

<div>
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=NSR4UkMAAAAJ" target="_blank">Zijian Chen</a><sup>1,2</sup>,
    Zhengyu Chen<sup>1</sup>, Bohan Liang<sup>2</sup>, Lirong Deng<sup>3</sup>, Yushuo Zheng<sup>1,2</sup>, Yanwei Jiang<sup>1,2</sup>, Qi Jia<sup>2</sup>, Kaiwei Zhang<sup>2</sup>,Wenjun Zhang<sup>1</sup>,
    <a href="https://ee.sjtu.edu.cn/FacultyDetail.aspx?id=24&infoid=66&flag=66" target="_blank">Guangtao Zhai</a><sup>1,2,*</sup>
</div>

<div>
  <sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>Shanghai AI Laboratory, <sup>3</sup>Macao Polytechnic University
</div>   

<div>
<sup>*</sup>Corresponding author. 
</div>


<div style="width: 80%; text-align:center; margin:auto;">
<img style="width:100%" src="asset/intro.png"></div>

</div>

> Abstract: Our benchmark offers (1) a comprehensive dataset and multi-dimensional evaluation pipeline; (2) an up-to-date leaderboard on MLLM figure reproduction proficiency; and (3) a nuanced understanding of the perceiving and reasoning bottlenecks hindering unified comprehension and generation in current MLLMs.


## News
- [TODO] 🔥 We are preparing the data and code.
- [2026/10/8] 🔥 [Github repo](https://github.com/zijianchen98/FigCodeBench) for **FigCodeBench** is online.


## Introduction
Researchers often have a scientific figure and need working code that reproduces it. For a model, this means turning visual structure into a rendered program, not merely describing the image or answering questions about it. We argue that text-only coding tests and image-understanding tests leave this perception-to-generation step unmeasured. This project has proposed a novel evaluation benchmark, aiming to assess the model's ability to observe complex charts (such as line graphs, bar charts, scatter plots, architecture diagrams, etc.) and generate corresponding reproducing code. This not only tests the model's visual perception and fine-grained understanding, but also significantly challenges its logical reasoning and alignment ability in generating code.

### Core Contributions
1. 📈 **Comprehensive**：Containing 6,194 image-code pairs, covering four programming languages (Python, Matlab, R, and Latex). 
2. 🛠️ **Automatic**：Providing image-oriented, code-oriented metrics, _Mean Machine Opinion Score_, and _Figure-Code Fidelity_ metric.
3. 🏆 **24 MLLMs**：In-depth experiments were conducted on 24 mainstream MLLMs, revealing the capability boundaries of different models in the "figure-to-code" task, as well as their adaptability to different programming languages and common error patterns.
---

## 📊 Dataset & Metrics

To be updated

* Huggingface: [downloading link]
* SJTU Netdisk: [downloading link]


### Environments
- Requirements:
To be updated


## Contact

Please contact the first author of this paper for queries.

- Zijian Chen, `zijian.chen@sjtu.edu.cn`

## Citation
Please feel free to cite our paper if you use the AGIN database in your research:
```
@misc{chen2026pixelcodingevaluatingfigure,
      title={From Pixel to Coding: Evaluating the Figure Reproduction Capabilities of MLLMs}, 
      author={Zijian Chen and Zhengyu Chen and Bohan Liang and Lirong Deng and Yushuo Zheng and Yanwei Jiang and Qi Jia and Kaiwei Zhang and Wenjun Zhang and Guangtao Zhai},
      year={2026},
      eprint={2610.10066},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.10066}, 
}
```

