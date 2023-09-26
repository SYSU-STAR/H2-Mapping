import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, data_path, use_gt=False,
                 scale_factor=0, crop=0, depth_scale=1000.0, max_depth=10, **kwargs) -> None:
        self.crop = crop
        self.depth_scale = depth_scale
        self.data_path = data_path
        self.scale_factor = scale_factor
        self.use_gt = use_gt
        num_imgs = len(glob(osp.join(data_path, 'color/*.jpg')))
        self.max_depth = max_depth
        self.K = self.load_intrinsic()
        self.depth_files = [
            osp.join(data_path, f'depth/{i}.png') for i in range(num_imgs)]
        self.image_files = [
            osp.join(data_path, f'color/{i}.jpg') for i in range(num_imgs)]
        self.pose_files = [
            osp.join(data_path, f'pose/{i}.txt') for i in range(num_imgs)]
        self.num_imgs = num_imgs

    def load_intrinsic(self):
        self.K = np.loadtxt(
            osp.join(self.data_path, 'intrinsic/intrinsic_depth.txt'))[:3, :3]
        if self.scale_factor > 0:
            scale = 2 ** self.scale_factor
            self.K = self.K / scale
            self.K[2, 2] = 1
        if self.crop > 0:
            self.K[0, 2] = self.K[0, 2] - self.crop
            self.K[1, 2] = self.K[1, 2] - self.crop
        return self.K

    def load_depth(self, index):
        depth = cv2.imread(self.depth_files[index], -1) / self.depth_scale
        depth[depth > self.max_depth] = 0
        if self.scale_factor > 0:
            skip = 2 ** self.scale_factor
            depth = depth[::skip, ::skip]
        if self.crop > 0:
            depth = depth[self.crop:-self.crop, self.crop:-self.crop]
        return depth

    def get_init_pose(self, init_frame=None):
        return np.loadtxt(self.pose_files[init_frame])

    def load_image(self, index):
        img = cv2.imread(self.image_files[index], -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480), cv2.INTER_AREA)  # / 255.0
        if self.scale_factor > 0:
            factor = 2 ** self.scale_factor
            size = (640 // factor, 480 // factor)
            img = cv2.resize(img, size, cv2.INTER_AREA)
        if self.crop > 0:
            img = img[self.crop:-self.crop, self.crop:-self.crop]
        return img / 255.0

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, index):
        img = torch.from_numpy(self.load_image(index)).float()
        depth = torch.from_numpy(self.load_depth(index)).float()
        pose = np.loadtxt(self.pose_files[index]) if self.use_gt else None
        return index, img, depth, self.K, pose


if __name__ == '__main__':
    import sys

    loader = DataLoader(sys.argv[1], 1)
    for data in loader:
        index, img, depth = data
        print(index, img.shape)
        cv2.imshow('img', img)
        cv2.waitKey(1)
