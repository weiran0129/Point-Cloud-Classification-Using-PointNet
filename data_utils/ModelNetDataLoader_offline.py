import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def compute_local_pca_features(xyz, k=16):
    """
    xyz: [N, 3]
    Returns per-point features:
        - curvature: [N, 1]
        - eigenvalues: [N, 3] (λ1 >= λ2 >= λ3)
        - density: [N, 1] (mean neighbor distance)
    """
    N = xyz.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)  # [N, k], [N, k]

    curvature = np.zeros((N, 1), dtype=np.float32)
    eigvals = np.zeros((N, 3), dtype=np.float32)
    density = np.zeros((N, 1), dtype=np.float32)

    eps = 1e-8
    for i in range(N):
        neigh = xyz[indices[i]]  # [k, 3]
        mu = neigh.mean(axis=0, keepdims=True)
        cov = (neigh - mu).T.dot(neigh - mu) / float(k)
        w, _ = np.linalg.eigh(cov)  # returns in ascending order
        w = np.sort(w)[::-1]        # descending: λ1 >= λ2 >= λ3
        eigvals[i] = w.astype(np.float32)

        s = w.sum() + eps
        curvature[i, 0] = float(w[-1] / s)   # λ3 / (λ1+λ2+λ3)

        density[i, 0] = distances[i].mean().astype(np.float32)

    return curvature, eigvals, density





class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        self.extra_features = set()
        if hasattr(args, 'extra_features') and args.extra_features:
            self.extra_features = set([s.strip() for s in args.extra_features.split(',') if s.strip()])

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = []
        for x in shape_ids[split]:
            # Remove .txt if present
            name = x.replace('.txt','')

            # Try matching against known class names
            matched = None
            for cname in self.classes.keys():  # e.g., night_stand, bathtub, etc.
                if name.startswith(cname):
                    matched = cname
                    break

            if matched is None:
                raise ValueError(f"ERROR: cannot determine class for {x}")

            shape_names.append(matched)
        if split == 'train':
            base_dir = self.root  # augmented ModelNet10
        else:
            base_dir = 'data/modelnet40_normal_resampled'  # original ModelNet40 test set

        self.datapath = [
            (shape_names[i], os.path.join(base_dir, shape_names[i], shape_ids[split][i]) + '.txt')
            for i in range(len(shape_ids[split]))
        ]
        
        self.datapath = [(cls, path.replace('\\', '/')) for cls, path in self.datapath]
        """
        print("\nDEBUG: First 10 datapath entries:")
        for i in range(min(10, len(self.datapath))):
            print("   ", self.datapath[i])
        print("Valid class names:", self.classes.keys(), "\n")
        """
        print('The size of %s data is %d' % (split, len(self.datapath)))
        
        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            key = self.datapath[index][0]
            if key not in self.classes:
                print("DEBUG ERROR: Invalid class key detected!")
                print("    index =", index)
                print("    datapath entry =", self.datapath[index])
                print("    key =", key)
                print("    valid classes =", list(self.classes.keys()))
                raise KeyError(key)   # raise again for stack trace

            cls = self.classes[key]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        # point_set: [N, C], first 3 dims are xyz, optionally normals in [3:6]
        xyz = point_set[:, 0:3]
        N = xyz.shape[0]

        extra_list = []

        # 1) radius: ||xyz||
        if 'radius' in self.extra_features:
            r = np.linalg.norm(xyz, axis=1, keepdims=True).astype(np.float32)  # [N, 1]
            extra_list.append(r)

        # 2) height: z - min(z)
        if 'height' in self.extra_features:
            z = xyz[:, 2:3]
            h = (z - z.min()).astype(np.float32)  # relative height
            extra_list.append(h)

        # 3) local PCA eigen + curvature + density
        if any(f in self.extra_features for f in ['pca', 'curvature', 'density']):
            curv, eigvals, dens = compute_local_pca_features(xyz, k=16)
            # pca eigenvalues
            if 'pca' in self.extra_features:
                extra_list.append(eigvals)        # [N, 3]
            if 'curvature' in self.extra_features:
                extra_list.append(curv)           # [N, 1]
            if 'density' in self.extra_features:
                extra_list.append(dens)           # [N, 1]

        if len(extra_list) > 0:
            extras = np.concatenate(extra_list, axis=1)   # [N, F_extra]
            point_set = np.concatenate([point_set, extras], axis=1)


        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)