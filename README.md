# ESD: Expected Squared Difference as a Tuning-Free Trainable Calibration Measure (ICLR 2023)

[[Paper](https://openreview.net/forum?id=bHW9njOSON)]

![](figure/esd.jpg)

## Requirements 
A suitable conda environment named esd can be created and activated with:
```
conda env create -f environment.yml
conda activate esd
```

## Prepare Datasets

Prepare ImageNet-100 dataset based on the following link https://github.com/danielchyeh/ImageNet-100-Pytorch

## Running Experiments
`script/baseline.sh`, `script/esd.sh`, `script/mmce.sh`, `script/sbece.sh` contain commands to run the baseline (NLL), NLL+ESD, NLL+MMCE, and NLL+SB-ECE, respectively. Change the path to the ImageNet-100 dataset in the bash files before running.

```
CUDA_VISIBLE_DEVICES=0 bash script/baseline.sh
```

## Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics), and Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments).

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{
yoon2023esd,
title={{ESD}: Expected Squared Difference as a Tuning-Free Trainable Calibration Measure},
author={Hee Suk Yoon and Joshua Tian Jin Tee and Eunseop Yoon and Sunjae Yoon and Gwangsu Kim and Yingzhen Li and Chang D. Yoo},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=bHW9njOSON}
}
```

## Contact
If you have any questions, please feel free to email hskyoon@kaist.ac.kr
