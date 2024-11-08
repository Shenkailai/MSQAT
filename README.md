# MSQAT: A Multi-dimension non-intrusive Speech Quality Assessment Transformer

Created by Kailai Shen, [Diqun Yan](http://www.yandiqun.com)†, Li Dong




## Prerequisite  

### Clone repository

```bash
git clone --recursive https://github.com/Shenkailai/MSQAT.git
```

### Dataset

Download links: [[ConferencingSpeech2022 Datasets]](https://github.com/ConferencingSpeech/ConferencingSpeech2022/tree/main/Training/Dev%20datasets) 

## Installation

### Local Environment
By `.yaml` file:

```bash
conda env create -f requirements.yaml
conda activate msqat
```

## Usage

## Experiments Structure



## Training and Evaluation Logs



## The Code Structure



## License

MIT License

## Acknowledgement



## Citation

If you find this codebase helpful, please consider to cite:

```
@article{SHEN2023109584,
title = {MSQAT: A multi-dimension non-intrusive speech quality assessment transformer utilizing self-supervised representations},
journal = {Applied Acoustics},
volume = {212},
pages = {109584},
year = {2023},
issn = {0003-682X},
doi = {https://doi.org/10.1016/j.apacoust.2023.109584},
url = {https://www.sciencedirect.com/science/article/pii/S0003682X23003821},
author = {Kailai Shen and Diqun Yan and Li Dong},
keywords = {Speech quality assessment, Non-intrusive, Self-supervised learning, Transformer},
abstract = {Convolutional neural networks (CNNs) have been widely utilized as the main building block for many non-intrusive speech quality assessment (NISQA) methods. A new trend is to add a self-attention mechanism based on CNN to better capture long-term global content. However, it is not clear whether the pure attention-based network is sufficient to obtain good performance in NISQA. To this end, a framework named Multi-dimension non-intrusive Speech Quality Assessment Transformer (MSQAT) is proposed. To strengthen the interactions of various speech regions between local and global, we proposed the Audio Spectrogram Transformer Block (ASTB), Transposed Attention Block (TAB) and the Residual Swin Transformer Block (RSTB). These three modules employ attention mechanisms across spatial and channel dimensions, respectively. Additionally, speech quality varies not only in different frames, but also in different frequencies. Thus, a two-branch structure is designed to better evaluate the quality of speech by considering the weighting of each patch's score. Experimental results demonstrate that the proposed MSQAT has state-of-the-art performance on three standard datasets (NISQA Corpus, Tencent Corpus, and PSTN Corpus) and indicate that the pure attention model can achieve or surpass the performance of other CNN-attention hybrid models.}
}
```
