import pickle
import numpy as np
import open3d as o3d
import random

data_path = r"C:\Users\wwr01\Pointnet_Pointnet2_pytorch\data\modelnet40_normal_resampled\modelnet40_train_1024pts.dat"

with open(data_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

if isinstance(data, dict):
    points = data['point']
    labels = data['label']
elif isinstance(data, (list, tuple)):
    points = data[0]
    labels = data[1]

# Pick random sample
i = random.randint(0, len(points)-1)
pc = points[i][:, :3]

print("Showing sample index:", i, "Label:", labels[i][0])

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([cloud])



