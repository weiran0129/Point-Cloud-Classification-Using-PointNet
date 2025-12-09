# Point-Cloud-Classification-Using-PointNet
## PointNet & PointNet++ Classification
PointNet and PointNet++ are foundational neural architectures designed to learn directly from unordered 3D point clouds. While PointNet captures global structural patterns, PointNet++ extends this capability by modeling hierarchical local geometry. In this project, we aim to improve classification accuracy on the ModelNet40 dataset by enhancing the original architectures through three key dimensions:
* **Data Augmentation:** `Dropout`, `Rotation`, `Scaling`, `Translation`, `Flipping`
* **Feature Engineering:** `Normals`, `Height`, `Radius`, `Curvature`, `Density`, `PCA`
* **Training Optimization:** `Cosine Aneealing`, `AdamW with Weight Decay`, `Label Smoothing`

## Environment Setup
This project was developed and tested on `Win10`, `PyTorch 2.5.1`, `CUDA 12.1`, `Python 3.9`.
```batch
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Data Preparation (Resampled ModelNet40)
[Resampled ModelNet40 Dataset (.txt file)](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled/data)

[Original ModelNet40 Dataset (.off file)](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset/code)

## Run
## Performance
## Visualization
We support a built-in point cloud visualizer for inspecting the resampled **ModelNet40** `.txt` point cloud files. 
```bash
pip install open3d numpy
python data_visualizer.py
```
<p align="center">
  <img src="visualizer/pic3.PNG" width="800">
</p>

## Referance
[PointNet](https://github.com/charlesq34/pointnet.git)

[PointNet2](https://github.com/charlesq34/pointnet2.git)

[Pytorch Version](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)

[Project Documents](https://drive.google.com/drive/folders/1otoBE9k_qX8wY8uKDhhJox0m2BNE7LU9?usp=drive_link)
