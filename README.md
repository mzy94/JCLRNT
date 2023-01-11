# JCLRNT
Code of CIKM'22 paper *Jointly Contrastive Learning on Road Network and Trajectory*.
An unsupervised method for road and trajectory representation utilizing contrastive learning.

## Experiment Results
### Results on road segment-based tasks

### Results on road trajectory-based tasks

## Usage
### Preparation
1. Download DiDi GAIA dataset from https://outreach.didichuxing.com/appEn-vue/dataList
2. Download and instal map matching tool from https://github.com/cyang-kth/fmm

### Preprocess
Run preprocessing to get map data and other features.
```
python data_processor.py
```
### Train and Evaluate
Train the model and evaluate it on different tasks
```
python main.py
```
