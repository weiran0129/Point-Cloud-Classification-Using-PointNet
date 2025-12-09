import os
import random
import numpy as np
import open3d as o3d

root = "data/modelnet40_normal_resampled"
grid_rows, grid_cols = 3, 3
num_items = grid_rows * grid_cols


classes = []
for d in os.listdir(root):
    dpath = os.path.join(root, d)
    if os.path.isdir(dpath) and any(f.endswith(".txt") for f in os.listdir(dpath)):
        classes.append(d)

if len(classes) < num_items:
    raise ValueError("Not enough valid classes with .txt files.")

chosen_classes = random.sample(classes, num_items)
pcds = []

for idx, cls in enumerate(chosen_classes):
    class_dir = os.path.join(root, cls)
    files = [f for f in os.listdir(class_dir) if f.endswith(".txt")]

    if len(files) == 0:
        continue  # skip empty folders

    chosen_file = random.choice(files)
    file_path = os.path.join(class_dir, chosen_file)

    # Load point cloud
    points = np.loadtxt(file_path, delimiter=",")
    xyz = points[:, :3]

    # Azure gradient
    z = xyz[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min())
    dark_blue  = np.array([0.05, 0.15, 0.35])
    light_blue = np.array([0.60, 0.80, 1.00])
    colors = (1 - z_norm)[:, None] * dark_blue + z_norm[:, None] * light_blue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Grid position
    row = idx // grid_cols
    col = idx % grid_cols
    pcd.translate([col * 2.0, -row * 2.0, 0])

    pcds.append(pcd)


if len(pcds) > 0:
    o3d.visualization.draw_geometries(pcds)
else:
    print("No point clouds loaded. Check dataset paths.")
