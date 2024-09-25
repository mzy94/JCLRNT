# JCLRNT
Code of CIKM'22 paper *Jointly Contrastive Learning on Road Network and Trajectory*.
An unsupervised method for road and trajectory representation utilizing contrastive learning.
![image](https://github.com/mzy94/JCLRNT/blob/main/pics/model.png)

## Experiment Results
### Results on road segment-based tasks
![image](https://github.com/mzy94/JCLRNT/blob/main/pics/road_task.png)

### Results on road trajectory-based tasks
![image](https://github.com/mzy94/JCLRNT/blob/main/pics/traj_task.png)

## Usage
### Preparation
1. Download DiDi GAIA dataset from https://outreach.didichuxing.com/appEn-vue/dataList (**Update Sept2024**: this link has been shut down. For the dataset, please refer to our drive: https://drive.google.com/drive/folders/1P_bSoUXNjifA3uxi8_IsRYco_H0UWyrv?usp=drive_link; We also provide the processed data for reproduction: https://drive.google.com/file/d/1WmdeNLEK7-otzQVleQmr5t_RZW-lZTZ0/view?usp=drive_link)
4. Download and instal map matching tool from https://github.com/cyang-kth/fmm

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
