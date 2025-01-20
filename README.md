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
<-- For more information about the project, please refer to our [project homepage](). -->
The article will be available soon...

## Prerequisites
Install all necessary packages using:

```shell
conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Data
Download ModelNet Dataset


## Train

Creatin Refined Dataset(ModeNet-R):
```shell
cd Point-SkipNet
bash creat_ModelNetR.sh
```


## Train

Train PointSkipNet from scratch using:
```shell
cd Point-SkipNet
bash run_train.sh
```

## Test

Test PointSkipNet with checkpoint using:
```shell
cd Point-SkipNet
bash run_train.sh
```


## Tree:
|<br />
|Ablation studies: <br />
|-- We performed ablation studies on two techniques;<br />
|--- Data augmentation:<br />
|------ No augmentation<br />
|------ All augmentations<br />
|------ anisotropic_scaling<br />
|------ jitter<br />
|------ rotation<br />
|------ translation<br />
|--- Skip Connection:<br />
|------ Concatenation<br />
|------ Plus<br />
|<br />
|-- Ablation studies were performed on three systems with three different graphics cards(Codes and log files of ablation studies are available in the following folders):<br />
|---- Nvidia GeForce GTX 1080: Although our goal was to study the PointSkipNet model on the new dataset i.e. Modelnet 40-2, but for the possibility of some interesting results, we also conducted studies on the old dataset i.e. Modelnet 40.<br />
|---- Nvidia GeForce GTX 2080: Conducting studies on the PointSkipNet model using the new dataset, Modelnet 40-2<br />
|---- Nvidia GeForce GTX 3080: Conducting studies on the PointSkipNet model using the new dataset, Modelnet 40-2<br />
|<br />
|<br />
|Confusion matrices_new dataset:<br />
|--- Confusion matrices of models trained using Modelnet40 are located in this folder.<br />
|<br />
|Confusion matrices_old dataset:<br />
|--- The confusion matrices of the models trained using Modelnet40-2 are located in this folder.<br />
|<br />
|Creating ModelNet40-2:<br />
|--- To create the Modelnet40-2 dataset, it is enough to put the "modelnet40_normal_resampled" version of the Modelnet40 dataset in this folder. By running the first two codes of this folder, the resampled version of the Modelnet40-2 dataset will be created, and if you run the third code, the "modelnet40_ply_hdf5_2048" version will also be created.<br />
|<br />
|Models: <br />
|--- CurveNet<br />
|--- DGCNN<br />
|--- Point-SkipNet<br />
|--- Point-NN<br />
|--- PointMLP<br />
|--- PointNet<br />
|--- PointNet++<br />
|<br />
|Point-SkipNet:<br />
| --- Our presented model is located in this folder. Despite the good performance, this model has competitive accuracy. (Codes, log file and pre-trained weights of this model are available in this folder)
<br />
<br />
### The results of the models trained on the Modelnet 2-40 dataset
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
| Point-SkipNet(ours)  | 94.33  | 92.93  | 2.9  |

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
