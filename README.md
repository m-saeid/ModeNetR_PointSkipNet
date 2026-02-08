# Enhancing 3D Point Cloud Classification with ModelNet‑R and Point‑SkipNet

<p align="center">
  <a href="https://www.arxiv.org/abs/2509.05198">
    <img src="https://img.shields.io/badge/Paper-arXiv-brightgreen" alt="Paper on arXiv"/>
  </a>
  <a href="https://m-saeid.github.io/ModeNetR_PointSkipNet">
    <img src="https://img.shields.io/badge/Project-Homepage-red" alt="Project Homepage"/>
  </a>
  <a href="https://www.youtube.com/watch?v=7ziipjpdth0&list=PLvWl5fdJgzQxaF0v4egv1cdrstl8N7fEM&index=2">
    <img src="https://img.shields.io/badge/Video-Presentation-blue" alt="YouTube Presentation"/>
  </a>
    <a href="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/ModelNet%E2%80%91R%20%26%20Point%E2%80%91SkipNet.pdf" target="_blank">
      <img src="https://img.shields.io/badge/Presentation-PDF-orange" alt="Presentation PDF"/>
    </a>
  <!--
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch Framework"/>
  </a>
  <a href="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"/>
  </a>
</p>
-->

## Overview

3D point cloud classification is pivotal in applications like autonomous driving, robotics, and augmented reality. However, existing benchmarks such as ModelNet40 suffer from inconsistent labeling, the presence of 2D data, size mismatches, and ambiguous class definitions.

**Our contributions:**
- **ModelNet‑R Dataset:** A refined version of ModelNet40 that resolves labeling inconsistencies, removes low-quality (almost 2D) data, adjusts size disparities, and sharpens class boundaries.
- **Point‑SkipNet:** A lightweight graph‑based neural network that leverages efficient sampling, neighborhood grouping, and skip connections to achieve state‑of‑the‑art classification accuracy with a reduced computational footprint.

For full details, please refer to the [paper on arXiv](https://www.arxiv.org/abs/2509.05198).

---

## Homepage
<!-- For more information about the project, please refer to our [project homepage](). -->
Official implementation of the paper "Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet",</p>
### accepted as an Oral presentation at IPRIA 2025 🔥
<br />The paper will be available soon...

---

## Problem Statement: Limitations of ModelNet40

<p align="center">
  <img src="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/ModelNet_Problems.jpg" width="500px" alt="Challenges in ModelNet40"/>
</p>

ModelNet40, while widely used, contains:
- **Inconsistent labeling:** Many samples are mislabeled or ambiguous.
- **2D artifacts:** A significant portion of the data is nearly flat, lacking volumetric depth.
- **Size mismatches:** Normalization often causes objects of different real-world scales to appear similar.
- **Poor class differentiation:** Similar geometries between classes (e.g., flower pot vs. vase) lead to confusion.

---

## Proposed Solution: ModelNet‑R & Point‑SkipNet

### Introducing ModelNet‑R
By addressing the issues of ModelNet40, ModelNet‑R provides a cleaner and more reliable benchmark for 3D point cloud research. It improves both the training process and the evaluation of classification models.

### Lightweight Classification with Point‑SkipNet
<p align="center">
  <img src="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/Point-SkipNet.jpg" width="500px" alt="Point‑SkipNet"/>
</p>

Point‑SkipNet is designed to:
- **Efficiently sample and group points** using a dedicated module.
- **Leverage skip connections** to preserve both global and local features.
- **Reduce computational overhead** while maintaining high classification accuracy.

### Architecture Overview
<p align="center">
  <img src="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/Point-SkipNet_Overview.jpg" width="500px" alt="Architecture Overview"/>
</p>

The architecture is modular:
- **Sample and Group Module:** Uses farthest point sampling (FPS) and ball queries to extract meaningful local neighborhoods.
- **Feature Aggregation:** Combines local geometric features with skip connections for robust global representation.

<p align="center">
  <img src="https://github.com/m-saeid/ModeNetR_PointSkipNet/blob/main/images/SampleAndGroup.jpg" width="500px" alt="Sample and Group Module"/>
</p>

---

## Experimental Insights

Experiments demonstrate that refining the dataset (ModelNet‑R) coupled with the efficient design of Point‑SkipNet results in significant performance improvements. For instance:

- **Point‑SkipNet on ModelNet:**  
  - Overall Accuracy (OA): ~92.29%  
  - Mean Class Accuracy (mAcc): ~89.84%

- **Point‑SkipNet on ModelNet‑R:**  
  - Overall Accuracy (OA): ~94.33%  
  - Mean Class Accuracy (mAcc): ~92.93%

These results underscore the importance of both dataset quality and computational efficiency.

---

## Prerequisites & Setup

Tested configurations include:

- **Configuration 1:**
  - OS: Ubuntu 22.04.3 LTS
  - GPU: NVIDIA GeForce RTX 2080 Ti
  - CUDA: 12.0
  - PyTorch: 2.1.1+cu121
  - Python: 3.10.12

- **Configuration 2:**
  - OS: Ubuntu 22.04.5 LTS
  - GPU: NVIDIA GeForce RTX 3080 Ti
  - CUDA: 12.2
  - PyTorch: 1.13.1+cu117
  - Python: 3.10.12

<!--```shell
conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```-->
---

## Data Preparation

Place the following datasets in the `data` folder:

- `data/modelnet40_normal_resampled`
- `data/modelnet40_ply_hdf5_2048`

> **Note:** The `modelnet40_normal_resampled` dataset will be converted to `modelnetR_normal_resampled` when you run the dataset creation script. It is recommended to back up the original data if needed.

---

## Creating ModelNet‑R

To generate the refined dataset:

```bash
cd Creating_ModelNet-R
h5=false bash creat_modelnetR.sh  # Use this if you do not require the h5 format
h5=true bash creat_modelnetR.sh   # Use this if you require the h5 format
```

Ensure that the `modelnet40_normal_resampled` dataset is present in the `data` folder before running the script.

---

## Training & Testing Point‑SkipNet

### Training

Train Point‑SkipNet on your chosen dataset:

```bash
cd Point-SkipNet
dataset="modelnetR" bash run_train.sh   # Train on ModelNet‑R (refined dataset)
dataset="modelnet" bash run_train.sh    # Train on ModelNet (original dataset)
```

### Testing

Test the trained model using the provided checkpoints:

```bash
cd Point-SkipNet
dataset="modelnetR" bash run_test.sh   # Test on ModelNet‑R
dataset="modelnet" bash run_test.sh    # Test on ModelNet
```

---

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
| Point-SkipNet(proposed)  | 94.33  | 92.93  | 1.47  |

<br />
<br />

## Citation

If you use this work, please cite:

```bibtex
@article{saeid2025enhancing,
  title={Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet},
  author={Saeid, Mohammad and Salarpour, Amir and MohajerAnsari, Pedram},
  journal={arXiv preprint arXiv:2509.05198},
  year={2025}
}
```

---

## Acknowledgements

This project builds upon the valuable contributions of the research community. Special thanks to:

- **PointNet++**  
  [Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/pdf/1706.02413) by Qi et al.

- **PointNet++ PyTorch Implementation**  
  [Repository by yanx27](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

- **ModelNet Dataset**  
  Originally presented in [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://openaccess.thecvf.com/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf)

We also acknowledge all related works cited in the paper for inspiring this study.

---

## Future Work

Future enhancements include:
- Refining all classes in ModelNet40 for uniform consistency.
- Incorporating techniques to retain size-related features lost during normalization.
- Extending validations of Point‑SkipNet to additional real‑world datasets such as ScanObjectNN and ShapeNet.

---

This repository offers an integrated approach to improve both dataset quality and model efficiency for 3D point cloud classification. Enjoy exploring the project, and feel free to reach out through the [project homepage](https://m-saeid.github.io/ModeNetR_PointSkipNet/) for any inquiries!
