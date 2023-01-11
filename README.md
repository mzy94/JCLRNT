# JCLRNT
Code of CIKM'22 paper *Jointly Contrastive Learning on Road Network and Trajectory*.
An unsupervised method for road and trajectory representation utilizing contrastive learning.

## Experiment Results
### Results on road segment-based tasks

### Results on road trajectory-based tasks

## Usage
### Dataset
Download DiDi GAIA dataset from https://outreach.didichuxing.com/appEn-vue/dataList

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
