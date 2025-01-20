import torch
import numpy as np

def Jitter(pcd, sigma=0.01, clip=0.05):
    npts, nfeats = pcd.shape
    jit_pts = np.clip(sigma * np.random.randn(npts, nfeats), -clip, clip)
    jit_pts += np.array(pcd, dtype=np.float32)
    return jit_pts

def Rotation(pcd, axis='y', angle=15):
    angle = np.random.uniform(-angle, angle)
    angle = np.pi * angle / 180
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]])
    elif axis == 'y':
        rotation_matrix = np.array([[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]])
    elif axis == 'z':
        rotation_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
    else:
        raise ValueError(f'axis should be one of x, y and z, but got {axis}!')
    rotated_pts = pcd @ rotation_matrix
    return rotated_pts

def Translation(pcd, shift=0.2):
    npts = pcd.shape[0]
    x_translation = np.random.uniform(-shift, shift)
    y_translation = np.random.uniform(-shift, shift)
    z_translation = np.random.uniform(-shift, shift)
    x = np.full(npts, x_translation)
    y = np.full(npts, y_translation)
    z = np.full(npts, z_translation)
    translation = np.stack([x, y, z], axis=-1)
    translation_pts = pcd + translation
    return translation_pts

def AnisotropicScaling(pcd, min_scale=0.66, max_scale=1.5):
    x_factor = np.random.uniform(min_scale, max_scale)
    y_factor = np.random.uniform(min_scale, max_scale)
    z_factor = np.random.uniform(min_scale, max_scale)
    scale_matrix = np.array([[x_factor, 0, 0], [0, y_factor, 0], [0, 0, z_factor]])
    scaled_pts = pcd @ scale_matrix
    return scaled_pts

def DataAugmentation(pcd, aug_type='all_aug', axis='y', angle=15, shift=0.2, min_scale=0.66, max_scale=1.5, sigma=0.01, clip=0.05):
    # aug = np.random.choice(['jitter', 'rotation', 'translation', 'anisotropic_scaling'])

    if aug_type == 'all_aug':
        aug = np.random.choice(['jitter', 'rotation', 'translation', 'anisotropic_scaling'])
    else:
        aug = aug_type
        
    if aug == 'jitter':
        pcd = Jitter(pcd, sigma, clip)
    elif aug == 'rotation':
        pcd = Rotation(pcd, axis, angle)
    elif aug == 'translation':
        pcd = Translation(pcd, shift)
    elif aug == 'anisotropic_scaling':
        pcd = AnisotropicScaling(pcd, min_scale, max_scale)

    np.random.shuffle(pcd)
    pcd = np.array(pcd, dtype=np.float32)

    return pcd


if __name__ == "__main__":
    import torch
    pcd = torch.rand(1024,3)
    for i in range(100):
        pcd = DataAugmentation(pcd)
        print(pcd.shape)