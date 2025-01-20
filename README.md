# Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet

<p>
<a href="https://arxiv.org/pdf/2302.14673.pdf">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a>
<a href="https://m-saeid.github.io/publications/ModelNet-R.html">
    <img src="https://img.shields.io/badge/Project-Homepage-red" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" /></a>
<a href="https://github.com/JunweiZheng93/APES/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
</p>

## Homepage
<!-- For more information about the project, please refer to our [project homepage](). -->
The paper will be available soon...

## Prerequisites
The latest codes are tested on:<br />
* Ubuntu 22.04.3 LTS, NVIDIA GeForce RTX 2080 Ti, CUDA 12.0, PyTorch 2.1.1+cu121 and Python 3.10.12<br />
* Ubuntu 22.04.5 LTS, NVIDIA GeForce RTX 3080 Ti, CUDA 12.2, PyTorch 1.13.1+cu117 and Python 3.10.12
<!--```shell
conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```-->

## Data
Place the following datasets in the data folder. 
* data/modelnet40_normal_resampled
* data/modelnet40_ply_hdf5_2048<br />
The "modelnet40_normal_resampled" dataset will be converted to "modelnetR_normal_resampled" and the h5 format of the dataset will be created after executing the code related to creating the dataset that follows.
If you need the information of the modelnet40_normal_resampled folder, be sure to make a backup copy. because the content of this folder will change and Modelnet40 will be converted to ModelnetR.


## Creating ModelNet-R
```shell
cd Creating ModelNet-R
h5=false bash creat_modelnetR.sh  # if you dont need h5 format of modelnet-R (To run Point-SkipNet code, it is not necessary to generate h5 format.)
h5=true bash creat_modelnetR.sh   # if you need h5 format of modelnet-R
```
The modelnet-R dataset is created using the modelnet40_normal_resampled dataset, so if you need to recreate the refined dataset(modelnet-R), you must put the modelnet40_normal_resampled dataset in the data folder.

## Train

Creating Refined Dataset(ModeNet-R):
```shell
cd Point-SkipNet
dataset="modelnetR" bash run_train.sh   # To train on the refined dataset(modeletR)
dataset="modelnet" bash run_train.sh    # To train on the original dataset(modelnet)
```


## Test

Test PointSkipNet with checkpoint using:
```shell
cd Point-SkipNet
dataset="modelnetR" bash run_test.sh   # To test on the refined dataset(modeletR)
dataset="modelnet" bash run_test.sh    # To test on the original dataset(modelnet)
```
* on the refined dataset(modeletR) >>> Test Instance Accuracy: 0.948759, Class Accuracy: 0.937396<br />
* on the original dataset(modelnet) >>> Test Instance Accuracy: 0.915210, Class Accuracy: 0.900735
<br />


### The results of the models trained on the ModelNet-R dataset
<br />

|     Model     |       OA      |      mAcc     |   Param(M)    |
| ------------- | ------------- | ------------- | ------------- |
| PointNet  | 91.39  | 88.79  | 3.47  |
| PointNet++ (ssg)  | 94.02  | 92.40  | 1.47  |
| PointNet++ (msg)  | 94.06  | 91.80  | 1.74  |
| Point-NN  | 84.75  | 77.65  | 0  |
| DG-CNN  | 94.03  | 92.64  | 1.8  |
| CurveNet  | 94.12  | 92.65  | 2.04  |
| PointMLP  | 95.33  | 94.30  | 12.6  |
| Point-SkipNet(proposed)  | 94.87  | 93.73  | 2.9  |

<br />
<br />

## Citation

If you are interested in this work, please cite as below:

```text
@modenetr{aaaa2025,
author={names},
title={Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet},
booktitle={Ipria},
year={2025}
}
```
