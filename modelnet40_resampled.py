import kagglehub
import shutil
import os

# 1. Download dataset
path = kagglehub.dataset_download("chenxaoyu/modelnet-normal-resampled")
print("Downloaded dataset path:", path)

# 2. The inner folder that you actually want
inner_folder = os.path.join(path, "modelnet40_normal_resampled")

# Check that it exists
if not os.path.exists(inner_folder):
    raise FileNotFoundError(f"Inner folder not found: {inner_folder}")

# 3. Destination = your project /data folder
dest = os.path.abspath(r"..\Point-Cloud-Classification-Using-PointNet\data")

os.makedirs(dest, exist_ok=True)

# 4. Final destination path
final_path = os.path.join(dest, "modelnet40_normal_resampled")

# 5. Move or copy ONLY the inner folder
shutil.copytree(inner_folder, final_path, dirs_exist_ok=True)

print("Dataset downloading complete, Final dataset path:", final_path)
