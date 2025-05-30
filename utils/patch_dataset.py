import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class PatchSRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=48, scale=4, augment=True, patches_per_image=10):
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith(('png', 'jpg'))])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('png', 'jpg'))])
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.patches_per_image = patches_per_image
        self.lr_patches, self.hr_patches = self.extract_patches()

    def extract_patches(self):
        lr_patches, hr_patches = [], []
        for lr_path, hr_path in zip(self.lr_files, self.hr_files):
            lr = cv2.imread(lr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            hr = cv2.imread(hr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

            h, w, _ = lr.shape
            lr_patch_size = self.patch_size // self.scale

            for _ in range(self.patches_per_image):
                top = random.randint(0, h - lr_patch_size)
                left = random.randint(0, w - lr_patch_size)
                lr_patch = lr[top:top+lr_patch_size, left:left+lr_patch_size, :]
                hr_patch = hr[top*self.scale:(top+lr_patch_size)*self.scale, left*self.scale:(left+lr_patch_size)*self.scale, :]

                if lr_patch.shape[:2] == (lr_patch_size, lr_patch_size) and hr_patch.shape[:2] == (self.patch_size, self.patch_size):
                    if self.augment:
                        if random.random() < 0.5:
                            lr_patch = np.fliplr(lr_patch).copy()
                            hr_patch = np.fliplr(hr_patch).copy()
                        if random.random() < 0.5:
                            lr_patch = np.flipud(lr_patch).copy()
                            hr_patch = np.flipud(hr_patch).copy()
                        if random.random() < 0.5:
                            k = random.randint(1, 3)
                            lr_patch = np.rot90(lr_patch, k).copy()
                            hr_patch = np.rot90(hr_patch, k).copy()

                    lr_patches.append(np.transpose(lr_patch, (2, 0, 1)))  # C,H,W
                    hr_patches.append(np.transpose(hr_patch, (2, 0, 1)))

        return lr_patches, hr_patches

    def __len__(self):
        return len(self.lr_patches)

    def __getitem__(self, idx):
        lr = torch.tensor(self.lr_patches[idx], dtype=torch.float32)
        hr = torch.tensor(self.hr_patches[idx], dtype=torch.float32)
        return lr, hr
