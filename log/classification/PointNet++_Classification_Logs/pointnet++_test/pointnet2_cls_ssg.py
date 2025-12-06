import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True, extra_channel=0):
        super(get_model, self).__init__()

        # original xyz (3) + optional normals (3) + extra features
        base_in_channel = 6 if normal_channel else 3
        in_channel = base_in_channel + extra_channel

        self.normal_channel = normal_channel
        self.extra_channel = extra_channel

        # Adjust the input channel for sa1
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,          # <-- UPDATED
            mlp=[64, 64, 128],
            group_all=False
        )

        # For subsequent layers:
        # SA1 outputs 128-dim features + xyz (3)
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False
        )

        # SA2 outputs 256-dim features + xyz (3)
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, C, N = xyz.shape
        
        # --- xyz always present ---
        xyz_only = xyz[:, :3, :]
    
        # --- collect feature channels ---
        feature_list = []
    
        if self.normal_channel:                 # normals = 3 channels
            feature_list.append(xyz[:, 3:6, :])
    
        if self.extra_channel > 0:              # extra features
            start = 3 + (3 if self.normal_channel else 0)
            feature_list.append(xyz[:, start:, :])
    
        points = torch.cat(feature_list, dim=1) if feature_list else None
        # ========================================================================================================================================================================
        if B == 1:  # or add custom debug flag
            print("\n====== DEBUG MODEL FORWARD ======")
            print("Input xyz shape:", xyz.shape)
            print("xyz_only shape:", xyz_only.shape)

            if self.normal_channel:
                print("Normal channels shape:", xyz[:, 3:6, :].shape)

            if self.extra_channel > 0:
                print("Extra feature channels shape:", xyz[:, start:, :].shape)

            print("Points passed to SA1:", None if points is None else points.shape)
            print("=================================\n")
        # ========================================================================================================================================================================
        # --- SA layers ---
        l1_xyz, l1_points = self.sa1(xyz_only, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
    
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
    
        return x, l3_points




class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss