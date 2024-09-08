from tqdm import tqdm
import numpy as np
import torch
import h5py
import os

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

labels = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox']

ds = "modelnet40_normal_resampled"

n_points = 2048
newpath = f'modelnet40_ply_hdf5_{n_points}' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

train_path = open(f'{ds}/modelnet40_train.txt')
train_path = train_path.read()
train_path = train_path.split('\n')
if train_path[-1] == '':
    train_path.pop(-1)

test_path = open(f'{ds}/modelnet40_test.txt')
test_path = test_path.read()
test_path = test_path.split('\n')
if test_path[-1] == '':
    test_path.pop(-1)

data = []
label = []

for mode in ['train', 'test']:
    r = 0
    paths = train_path if mode == 'train' else test_path
    for i in tqdm(range(len(paths)), desc = f'{mode}_data'):
        p = open(f'{ds}/{paths[i][:-5]}/{paths[i]}.txt')
        p = p.read()
        p = p.split('\n')

        xyz = []
        for xyz_nrmls in p:
            p = xyz_nrmls.split(',')[:3]
            if p != ['']:
                p = [float(p[0]), float(p[1]), float(p[2])]
                xyz.append(p)

        data.append(xyz)
        label.append(labels.index(paths[i][:-5]))

        if len(data) == 2048 or (mode=='train' and r==4 and len(paths)==i+1) or (mode=='test' and r==1 and len(paths)==i+1):
            data = np.array(data)
            label = np.array(label)

            #FPS
            print(f'FPS was started - mode: {mode}  -  h5_number: {r}')
            data = torch.tensor(data).to(device)
            fps_idx = farthest_point_sample(data, n_points).long()  # [B, npoint]
            data = index_points(data, fps_idx)  # [B, npoint, 3]
            data = data.cpu().numpy()
            #data = data[:,n_points,:]

            with h5py.File(f'modelnet40_ply_hdf5_{n_points}/ply_data_{mode}{r}.h5', 'w') as f:
                f.create_dataset('data', data = data)
                f.create_dataset('label', data = label)
            r+=1
            data = []
            label = []

