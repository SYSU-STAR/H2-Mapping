import math

import torch


def back_project(centers, c2w, K, depth, truncation):
    """
    Back-project the center point of the voxel into the depth map.
    Transform the obtained depth in the camera coordinate system to the world
    
    Args:
        centers (tensor, num_voxels*3): voxel centers
        c2w (tensor, 4*4): camera coordinate to world coordinate.
        K (array, 3*3): camera reference
        depth (tensor, w*h): depth ground true.
        truncation (float): truncation value.
    Returns:
        initsdf (tensor,num_voxels): Each vertex of the voxel corresponds to the depth value of the depth map,
                                          if it exceeds the boundary, it will be 0
        seen_iter_mask (tensor,num_voxels): True if two points match
        (1).The voxel is mapped to the corresponding pixel in the image, and does not exceed the image boundary
        (2).The initialized sdf value should be within the cutoff distance.
    """
    H, W = depth.shape
    w2c = torch.linalg.inv(c2w.float())
    K = torch.from_numpy(K).cuda()
    ones = torch.ones_like(centers[:, 0]).reshape(-1, 1).float()
    homo_points = torch.cat([centers, ones], dim=-1).unsqueeze(-1).float()
    homo_cam_points = w2c @ homo_points  # (N,4,1) = (4,4) * (N,4,1)
    cam_points = homo_cam_points[:, :3]  # (N,3,1)
    uv = K.float() @ cam_points.float()
    z = uv[:, -1:] + 1e-8
    uv = uv[:, :2] / z  # (N,2)
    uv = uv.round()
    cur_mask_seen = (uv[:, 0] < W) & (uv[:, 0] > 0) & (uv[:, 1] < H) & (uv[:, 1] > 0)
    cur_mask_seen = (cur_mask_seen & (z[:, :, 0] > 0)).reshape(-1)  # (N_mask,1) -> (N_mask)
    uv = (uv[cur_mask_seen].int()).squeeze(-1)  # (N_mask,2)
    depth = depth.transpose(-1, -2)  # (W,H)

    initsdf = torch.zeros((centers.shape[0], 1), device=centers.device)

    voxel_depth = torch.index_select(depth, dim=0, index=uv[:, 0]).gather(dim=1, index=uv[:, 1].reshape(-1,
                                                                                                        1).long())  # (N_mask,1)

    initsdf[cur_mask_seen] = (voxel_depth - cam_points[cur_mask_seen][:, 2]) / truncation  # (N,1)
    seen_iter_mask = cur_mask_seen

    return initsdf.squeeze(-1), seen_iter_mask


@torch.no_grad()
def initemb_sdf(frame, map_states, truncation, voxel_size=None, octant_idx=None, voxel_initialized=None,
                vertex_initialized=None, use_gt=True):
    vertexes = map_states["voxel_vertex_idx"]
    centers = map_states["voxel_center_xyz"]

    sdf_priors = map_states["sdf_priors"]

    novertexes_mask = ~(vertexes.eq(-1).any(-1))

    depth = frame.depth
    K = frame.K  # (3,3)
    if use_gt:
        c2w = frame.get_ref_pose().cuda()  # (4,4)
    else:
        c2w = frame.get_ref_pose().cuda() @ frame.get_d_pose().cuda()  # (4,4)

    octant_idx = octant_idx[novertexes_mask][:, 0]
    uninit_idx = ~ voxel_initialized[octant_idx.long()]
    centers = centers[novertexes_mask][uninit_idx]  # （N，3）
    vertexes = vertexes[novertexes_mask, :][uninit_idx]  # (N,8)

    """
    vert_cord relative to voxel_cord: 
            [[-1., -1., -1.],
            [-1., -1.,  1.],
            [-1.,  1., -1.],
            [-1.,  1.,  1.],
            [ 1., -1., -1.],
            [ 1., -1.,  1.],
            [ 1.,  1., -1.],
            [ 1.,  1.,  1.]]
    """
    cut_x = cut_y = cut_z = torch.linspace(-1, 1, 2)
    cut_xx, cut_yy, cut_zz = torch.meshgrid(cut_x, cut_y, cut_z, indexing='ij')
    offsets = torch.stack([cut_xx, cut_yy, cut_zz], dim=-1).int().reshape(-1, 3).to(centers.device)

    occ_mask = torch.ones(centers.shape[0]).to(centers.device).bool()

    centers_vert = (centers.unsqueeze(1) + offsets * (voxel_size / 2)).reshape(-1, 3)
    initsdf_vert, seen_iter_mask_vert = back_project(centers_vert, c2w, K, depth, truncation)
    occ_mask[((initsdf_vert.reshape(-1, 8) * truncation).abs() > math.sqrt(6) * voxel_size).any(-1)] = False
    occ_mask = occ_mask * (seen_iter_mask_vert.reshape(-1, 8)).all(-1)

    mask = ~ vertex_initialized[vertexes[occ_mask].to(torch.long).reshape(-1)].clone()
    sdf_priors[vertexes[occ_mask].reshape(-1)[mask].to(torch.long), 0] = \
    initsdf_vert.reshape(-1, 8)[occ_mask].reshape(-1)[mask].clone()
    vertex_initialized[vertexes[occ_mask].reshape(-1)[mask].to(torch.long)] += True

    voxel_initialized[octant_idx[uninit_idx][occ_mask].long()] += True

    map_states["sdf_priors"] = sdf_priors

    return map_states, voxel_initialized, vertex_initialized
