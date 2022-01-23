# Introduction

This is the implementation of the paper [SMoA: Searching a Modality-Oriented Architecture for Infrared and Visible Image Fusion](https://ieeexplore.ieee.org/abstract/document/9528046).

## Requirements

* python >= 3.6
* pytorch == 1.7
* torchvision == 0.8

## Datasets

You can download the datasets [here](https://pan.baidu.com/s/1kUja4iau37MwLnGI8_lMWg?pwd=eapv).

## Test

```shell
python test.py
```

## Train from scratch

### step 1

```shell
python train_search.py
```

### step 2

Find the string which descripting the searched architectures in the log file. Copy and paste it into the genotypes.py, the format should consist with the primary architecture string.

### step 3

```shell
python train.py
```

## Citation

If you use any part of this code in your research, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/9528046):

```
@ARTICLE{9528046,
author={Liu, Jinyuan and Wu, Yuhui and Huang, Zhanbo and Liu, Risheng and Fan, Xin},
journal={IEEE Signal Processing Letters},
title={SMoA: Searching a Modality-Oriented Architecture for Infrared and Visible Image Fusion},
year={2021},
volume={28},
number={},
pages={1818-1822},
doi={10.1109/LSP.2021.3109818}}
```
