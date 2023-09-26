import numpy as np
import torch.nn as nn

from se3pose import OptimizablePose
from utils.sample_util import *

rays_dir = None


class RGBDFrame(nn.Module):
    def __init__(self, fid, rgb, depth, K, offset, ref_pose=None) -> None:
        super().__init__()
        self.stamp = fid
        self.h, self.w = depth.shape
        if type(rgb) != torch.Tensor:
            rgb = torch.FloatTensor(rgb).cuda()
        if type(depth) != torch.Tensor:
            depth = torch.FloatTensor(depth).cuda()  # / 2
        self.rgb = rgb.cuda()
        self.depth = depth.cuda()
        self.K = K

        if ref_pose is not None:
            if len(ref_pose.shape) != 2:
                ref_pose = ref_pose.reshape(4, 4)
            if type(ref_pose) != torch.Tensor:  # from gt data
                self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
                self.ref_pose[:3, 3] += offset  # Offset ensures voxel coordinates>0
            else:  # from tracked data
                self.ref_pose = ref_pose.clone().requires_grad_(False)
            self.d_pose = OptimizablePose.from_matrix(torch.eye(4, requires_grad=False, dtype=torch.float32))
        else:
            self.ref_pose = None
        self.precompute()

    def get_d_pose(self):
        return self.d_pose.matrix()

    def get_d_translation(self):
        return self.d_pose.translation()

    def get_d_rotation(self):
        return self.d_pose.rotation()

    def get_d_pose_param(self):
        return self.d_pose.parameters()

    def get_ref_pose(self):
        return self.ref_pose

    def get_ref_translation(self):
        return self.ref_pose[:3, 3]

    def get_ref_rotation(self):
        return self.ref_pose[:3, :3]

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None):
        w = self.w if w == None else w
        h = self.h if h == None else h
        if K is None:
            K = np.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(
            torch.arange(w), torch.arange(h), indexing='xy')
        rays_d = torch.stack(
            [(ix - K[0, 2]) / K[0, 0],
             (iy - K[1, 2]) / K[1, 1],
             torch.ones_like(ix)], -1).float()  # camera coordinate
        return rays_d

    @torch.no_grad()
    def precompute(self):
        global rays_dir
        if rays_dir is None:
            rays_dir = self.get_rays(K=self.K).cuda()
        self.rays_d = rays_dir
        self.points = self.rays_d * self.depth[..., None]
        self.valid_mask = self.depth > 0

    @torch.no_grad()
    def get_points(self):
        return self.points[self.valid_mask].reshape(-1, 3)  # [N,3]

    @torch.no_grad()
    def sample_rays(self, N_rays):
        self.sample_mask = sample_rays(
            torch.ones_like(self.depth)[None, ...], N_rays)[0, ...]
