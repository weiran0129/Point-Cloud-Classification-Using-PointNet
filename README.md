# Point-Cloud-Classification
## PointNet & PointNet++ Classification
PointNet and PointNet++ are foundational neural architectures designed to learn directly from unordered 3D point clouds. While PointNet captures global structural patterns, PointNet++ extends this capability by modeling hierarchical local geometry. In this project, we aim to improve classification accuracy on the ModelNet40 dataset by enhancing the original architectures through three key dimensions:
* **Data Augmentation:** `Dropout`, `Rotation`, `Scaling`, `Translation`, `Flipping`
* **Feature Engineering:** `Normals`, `Height`, `Radius`, `Curvature`, `Density`, `PCA`
* **Training Optimization:** `Cosine Aneealing`, `AdamW with Weight Decay`, `Label Smoothing`

## Environment Setup
Clone the Repo
```batch
git clone https://github.com/weiran0129/Point-Cloud-Classification-Using-PointNet.git
cd Point-Cloud-Classification-Using-PointNet
```

This project was developed and tested on `Win10`, `PyTorch 2.5.1`, `CUDA 12.1`, `Python 3.9`.
```batch
conda create -n PointCloudClassification python=3.9
conda activate PointCloudClassification
conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install required python packages.

## Data Preparation (Resampled ModelNet40)
This project uses the ModelNet40 Normal Resampled dataset (`.txt` format), a standardized, preprocessed version of the original ModelNet40.
* Uniform 1,024 points per shape with train/test spilt.
* Normalized coordinates.
* Per-point surface normals.

Download to `data/` by runing
```batch
pip install kagglehub
python download_dataset.py
```
[Resampled ModelNet40 Dataset (.txt file)](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled/data) | [Original ModelNet40 Dataset (.off file)](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset/code)
## Run
This project applies both online and offline data augmentation to improve generalization on ModelNet10/40.
* **On-fly Data Augmentation:** Applied dynamically during each forward pass. The batch of point-cloud data is transformed in memory without increasing the number of samples in the dataset. Aim for efficiency. (Enabled using `train_classification.py`)
* **Off-line Data Augmentation:** Creates an augmented copy of the dataset on disk. By default, 50% of the training samples are duplicated and augmented, this increases both dataset diversity and dataset size. (Enabled using `train_offline.py`)
  * Auto deletion of the created augmented dataset for space saving. Comment out for permantly save.
```bash
if args.offline_aug:
  aug_dir = os.path.join('data', f"modelnet10_aug_{args.aug_ops.replace(',', '_')}")
  if os.path.exists(aug_dir):
    print(f"[INFO] Deleting offline augmented dataset: {aug_dir}")
    shutil.rmtree(aug_dir)
```
  
Below are some general arguments use to train the model.
| Argument        | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `--model` | pointnet_cls | Model to load (`pointnet_cls`, `pointnet2_cls_ssg`, etc.) |
| `--num_category` | 40 | Switch between ModelNet40 and ModelNet10 |
| `--batch_size`  | 24      | Batch size                           |
| `--epoch`       | 75      | Number of training epochs            |
| `--learning_rate` | 0.001 | Initial learning rate                |
| `--use_normals` | False | Include surface normals in input |
| `--aug_ops` | None | Data Augmentation methods: `dropout`, `jitter`, `scale`, etc. |
| `--extra_features` | None | Feature Engineering: `height`, `radius`, `curvature`, etc. |
| `--optimizer`   | Adam    | Optimizer: `Adam`, `AdamW`, or `SGD` |
| `--log_dir`     | None    | Directory to save logs/checkpoints   |
| `--offline_aug` | None | Perform offline augmentation before training |

Example Training
```batch
## Baseline PointNet
python train_classification.py --log_dir PointNet

## PointNet++ with augmentation, feature engineering, and optimization
python train_classification.py \
    --model pointnet2_cls_ssg \
    --aug_ops "dropout,jitter_sigma_clip,scale" \
    --extra_features "height,radius,density" \
    --use_normals \
    -- optimizer AdamW \
    --log_dir PointNet++
```
## Performance
Compare to baseline (No data augmentation, feature engineering, optimization), we have

* Best PointNet: `Dropout+Jitter+Normals+Height+Radius` | `InstAcc` **0.917 (+3.8%)** | `ClassAcc` **0.884 (+5.7 %)**

* Best PointNet++: `Dropout+Scale+Jitter+Normals with Optimization` | `InstAcc` **0.935 (+2.1%)** | `ClassAcc` **0.911 (+2.8 %)**

Explore you own combination.
## Visualization
We support a built-in point cloud visualizer for inspecting the resampled **ModelNet40** `.txt` point cloud files. 
```bash
pip install open3d numpy
python data_visualizer.py
```
<p align="center">
  <img src="visualizer/pic3.PNG" width="800">
</p>

## Project Files
[Report](https://drive.google.com/drive/folders/1otoBE9k_qX8wY8uKDhhJox0m2BNE7LU9?usp=drive_link) | [Presentation Slide](https://drive.google.com/drive/folders/1otoBE9k_qX8wY8uKDhhJox0m2BNE7LU9?usp=drive_link)| [Related Documents](https://drive.google.com/drive/folders/1otoBE9k_qX8wY8uKDhhJox0m2BNE7LU9?usp=drive_link)

## Referance
[PointNet](https://github.com/charlesq34/pointnet.git) | [PointNet2](https://github.com/charlesq34/pointnet2.git) | [Pytorch Version](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)
