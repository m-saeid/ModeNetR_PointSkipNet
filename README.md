# Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet

<p align="center">
  <a href="https://arxiv.org/link_paper">
    <img src="https://img.shields.io/badge/Paper-arXiv-brightgreen" alt="Paper on arXiv"/>
  </a>
  <a href="https://github.com/m-saeid/ModeNetR_PointSkipNet/">
    <img src="https://img.shields.io/badge/Project-Homepage-red" alt="Project Homepage"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch Framework"/>
  </a>
  <a href="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"/>
  </a>
  <a href="https://www.youtube.com/watch?v=7ziipjpdth0&list=PLvWl5fdJgzQxaF0v4egv1cdrstl8N7fEM&index=2">
    <img src="https://img.shields.io/badge/Video-Presentation-blue" alt="YouTube Presentation"/>
  </a>
</p>

## Overview

This repository contains the official implementation of the paper **"Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet,"** accepted for oral presentation at **IPRIA 2025**. The paper will be available soon on [arXiv](https://arxiv.org/link_paper).

### Key Challenges in ModelNet40

![ModelNet Problems](https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/ModelNet_Problems.jpg)

The performance of nearly all methods on ModelNet40 lacks consistency. Running the same code multiple times can yield varying results, even with a fixed seed. More details are discussed in [this issue](https://github.com/CVMI-Lab/PAConv/issues/9#issuecomment-873371422).

### Proposed Solution: Point-SkipNet

![Point-SkipNet](https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/Point-SkipNet.jpg)

Our proposed **Point-SkipNet** enhances classification accuracy by leveraging a lightweight positional encoding mechanism. This significantly improves performance on the **ModelNet-R** dataset, a refined version of ModelNet40.

### Architecture Overview

![Point-SkipNet Overview](https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/Point-SkipNet_Overview.jpg)

## Prerequisites

The code has been tested on the following configurations:

- **Configuration 1**:
  - OS: Ubuntu 22.04.3 LTS
  - GPU: NVIDIA GeForce RTX 2080 Ti
  - CUDA: 12.0
  - PyTorch: 2.1.1+cu121
  - Python: 3.10.12

- **Configuration 2**:
  - OS: Ubuntu 22.04.5 LTS
  - GPU: NVIDIA GeForce RTX 3080 Ti
  - CUDA: 12.2
  - PyTorch: 1.13.1+cu117
  - Python: 3.10.12

## Data Preparation

Place the following datasets in the `data` folder:

- `data/modelnet40_normal_resampled`
- `data/modelnet40_ply_hdf5_2048`

**Note**: The `modelnet40_normal_resampled` dataset will be converted to `modelnetR_normal_resampled` after executing the dataset creation script. If you need the original `modelnet40_normal_resampled` data, please make a backup, as its content will change during the conversion to ModelNet-R.

### Sample and Grouping Strategy

![Sample and Group](https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/SampleAndGroup.jpg)

## Creating ModelNet-R

To create the ModelNet-R dataset:

```bash
cd Creating_ModelNet-R
h5=false bash creat_modelnetR.sh  # If you don't need the h5 format of ModelNet-R
h5=true bash creat_modelnetR.sh   # If you need the h5 format of ModelNet-R
```

**Note**: The ModelNet-R dataset is created using the `modelnet40_normal_resampled` dataset. Ensure this dataset is in the `data` folder before running the script.

## Training Point-SkipNet

To train Point-SkipNet from scratch:

```bash
cd Point-SkipNet
dataset="modelnetR" bash run_train.sh   # To train on the refined dataset (ModelNet-R)
dataset="modelnet" bash run_train.sh    # To train on the original dataset (ModelNet)
```

## Testing Point-SkipNet

To test Point-SkipNet with a checkpoint:

```bash
cd Point-SkipNet
dataset="modelnetR" bash run_test.sh   # To test on the refined dataset (ModelNet-R)
dataset="modelnet" bash run_test.sh    # To test on the original dataset (ModelNet)
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{your2025paper,
  author    = {Your Name and Co-author Name},
  title     = {Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet},
  booktitle = {Proceedings of the IPRIA 2025},
  year      = {2025},
}
```

## Acknowledgements

This project builds upon the foundation of excellent prior work from the research community. Key contributions include:

- **PointNet++**: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413) by Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas
- **Implementation**: [PointNet++ PyTorch Implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) by yanx27
- **Dataset**: Modifications to the [ModelNet40 dataset](https://modelnet.cs.princeton.edu/), originally presented in [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://openaccess.thecvf.com/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf) by Wu et al.

We recognize and appreciate the efforts of these original authors and contributors, without whom this project would not have been possible.

