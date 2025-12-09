# Point-Cloud-Classification-Using-PointNet
## PointNet & PointNet++ Classification
Compare the performance of PointNet &amp; PointNet++, explore their ability to generalize across different object categories, and investigate techniques to enhance classification accuracy
[Project Documents](https://drive.google.com/drive/folders/1otoBE9k_qX8wY8uKDhhJox0m2BNE7LU9?usp=drive_link)
### Data Augmentation
### Feature Engineering
### Training Optimization

## Environment Setup
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
