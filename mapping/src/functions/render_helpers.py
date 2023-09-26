import torch
import torch.nn.functional as F

from .voxel_helpers import ray_intersect, ray_sample


def ray(ray_start, ray_dir, depths):
    """
    Calculate the coordinates of the sampling point in the world coordinate system according to the light origin, 
    direction and depth value of the sampling point
    
    Args:
        ray_start (tensor, N_rays*1*3): ray's origin coordinates in world coordinate.
        ray_dir (tensor, N_rays*1*3): ray's dir in world coordinate.
        depths (tensor, N_rays*N_points*1): depths of sampling points along the ray.

    Returns:
        ray_start+ray_dir*depth (tensor, N_rays*N_points*3): sampling points in world
    """
    return ray_start + ray_dir * depths


def fill_in(shape, mask, input, initial=1.0):
    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    return output.masked_scatter(mask.unsqueeze(-1).expand(*shape), input)


def masked_scatter(mask, x):
    """
    The sampling points that did not hit the voxel were masked in the previous program, 
    this function restores the previous dimension, and the masked element is set to 0
    
    Args:
        mask (tensor, N_rays*N_points_every_ray):
        x (tensor, samples_points):
    
    Returns:
        (tensor, N_rays*n_points_every_ray):Restore the dimension before the mask, the position of the unimpacted voxel is 0
    """
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


def masked_scatter_ones(mask, x):
    """
    The sampling points that did not hit the voxel were masked in the previous program, 
    this function restores the previous dimension, and the masked element is set to 1
    
    Args:
        mask (tensor, N_rays*N_points_every_ray):
        x (tensor, samples_points):
    
    Returns:
        (tensor, N_rays*n_points_every_ray):Restore the dimension before the mask, the position of the unimpacted voxel is 1
    """
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_ones(B, K).masked_scatter(mask, x)
    return x.new_ones(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


@torch.enable_grad()
def trilinear_interp(p, q, point_feats):
    """
    For the feature vector stored in a voxel, 
    perform cubic linear interpolation to obtain the feature of each point.
    """
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
    if point_feats.dim() == 2:
        point_feats = point_feats.view(point_feats.size(0), 8, -1)

    point_feats = (weights * point_feats).sum(1)
    return point_feats


def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
    c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
    ox, oy, oz = torch.meshgrid([c, c, c], indexing='ij')
    offset = (torch.cat([
        ox.reshape(-1, 1),
        oy.reshape(-1, 1),
        oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    if not offset_only:
        return (
                point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel)
    return offset.type_as(point_xyz) * quarter_voxel  # (8,3)


@torch.enable_grad()
def get_embeddings(sampled_xyz, point_xyz, point_feats, voxel_size):
    """
    run cubic linear interrpolation and get features corresponding to sampling points.
    
    Args:
        sampled_xyz (tensor, N_points*3): points x,y,z which belong to it voxel
        point_xyz (tensor, N_points*3): voxel center x,y,z
        point_feats (tensor, N_points, 8*embed_dim): features of sample point vertices 
        voxel_size (int): voxel size
    
    Returns:
        feats (tensor, N_points*embed_dim): features after cubic linear interpolation.
    """
    # tri-linear interpolation
    p = ((sampled_xyz - point_xyz) / voxel_size + 0.5).unsqueeze(1)  # add 0.5 value are clamped to [0,1]
    q = offset_points(p, 0.5, offset_only=True).unsqueeze(0) + 0.5  # range[-0.5,0.5] + 0.5 shape:(1,8,3)
    feats = trilinear_interp(p, q, point_feats).float()  # (N_points,32)
    return feats


@torch.enable_grad()
def get_features(samples, map_states, voxel_size):
    """
    Retrieve the voxel corresponding to the sampling point and the surrounding vertices, 
    and obtain the input features of each sampling point through cubic linear interpolation
    
    Args:
        samples (dict): sampling points information.
        map_states (dict): voxel information according to the octrees.
        voxel_size (int): voxel size.
    
    Returns:
        inputs (dict): sampled distance(N_points) and embedding feature vectors(N_points,emb_dim).
    """
    # encoder states
    point_feats = map_states["voxel_vertex_idx"].cuda()
    point_xyz = map_states["voxel_center_xyz"].cuda()  # (voxel_num,3)
    sdf_priors_all = map_states["sdf_priors"].cuda()

    # ray point samples
    sampled_idx = samples["sampled_point_voxel_idx"].long()
    sampled_xyz = samples["sampled_point_xyz"].requires_grad_(True)
    # sampled_idx stores the index of each sampling point corresponding to voxel_center_xyz, 
    # and after F.embedding, the voxel center corresponding to each point is obtained
    point_xyz = F.embedding(sampled_idx, point_xyz)  # (chunk_size,3)
    # sampled_idx is the voxel id corresponding to the sampling point, 
    # and find the vert id contained in feats according to the voxel id
    point_emd_idx = F.embedding(sampled_idx, point_feats)  # (chunk_size,8)

    point_sdf_priors = F.embedding(point_emd_idx, sdf_priors_all).view(point_xyz.size(0),
                                                                       -1)  # (chunk_size,8,emd_dim) -> (chunk_size,8*emd_dim)
    sdf_priors = get_embeddings(sampled_xyz, point_xyz, point_sdf_priors, voxel_size)
    feats = None

    inputs = {"emb": feats, "sdf_priors": sdf_priors}
    return inputs


@torch.no_grad()
def get_scores(sdf_network, map_states, voxel_size, bits=8, model="parallel_hash_net"):
    """
    This function is used in the get_mesh process to obtain the sdf value of each sampling point.
    """
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    sdf_priors = map_states["sdf_priors"]

    chunk_size = 32
    res = bits

    @torch.no_grad()
    def get_scores_once(feats, points):
        # sample points inside voxels
        start = -.5
        end = .5  # - 1./bits

        x = y = z = torch.linspace(start, end, res)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float().cuda()

        sampled_xyz *= voxel_size
        sampled_xyz = sampled_xyz.reshape(1, -1, 3) + points.unsqueeze(1)

        sampled_idx = torch.arange(points.size(0), device=points.device)
        sampled_idx = sampled_idx[:, None].expand(*sampled_xyz.size()[:2])
        sampled_idx = sampled_idx.reshape(-1)
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        if sampled_xyz.shape[0] == 0:
            return

        field_inputs = get_features(
            {
                "sampled_point_xyz": sampled_xyz,
                "sampled_point_voxel_idx": sampled_idx,
                "sampled_point_ray_direction": None,
                "sampled_point_distance": None,
            },
            {
                "voxel_vertex_idx": feats,
                "voxel_center_xyz": points,
                "sdf_priors": sdf_priors,
            },
            voxel_size
        )

        sdf_values = sdf_network.get_sdf(sampled_xyz)
        sdf_values = sdf_values[:, -1] + field_inputs['sdf_priors'][:, -1].float().cuda()

        return sdf_values.reshape(-1, res ** 3, 1).detach().cpu()

    return torch.cat([
        get_scores_once(feats[i: i + chunk_size],
                        points[i: i + chunk_size])
        for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 1)


@torch.no_grad()
def eval_points(sdf_network, sampled_xyz):
    def get_scores_once(sampled_xyz):
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        if sampled_xyz.shape[0] == 0:
            return
        color = sdf_network.get_color(sampled_xyz)
        return color.detach().cpu()

    chunk_size = 3200

    results = []
    for i in range(0, sampled_xyz.size(0), chunk_size):
        score_once = get_scores_once(sampled_xyz[i: i + chunk_size].cuda())
        results.append(score_once)
    results = torch.cat(results, dim=0)
    return results


# convert sdf to weight
def sdf2weights(sdf_in, trunc, z_vals, sample_mask_per):
    weights = torch.sigmoid(sdf_in / trunc) * \
              torch.sigmoid(-sdf_in / trunc)
    # use the change of sign to find the surface, sdf's sign changes as it cross the surface
    signs = sdf_in[:, 1:] * sdf_in[:, :-1]
    mask = torch.where(
        signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs)
    )
    # return the index of the closest point outside the surface
    inds = torch.argmax(mask, axis=1)
    inds = inds[..., None]
    z_min = torch.gather(z_vals, 1, inds)
    # calculate truncation mask, delete the point behind the surface and exceed trunc, z_min is here approximate the surface
    mask = torch.where(
        z_vals < z_min + trunc,
        torch.ones_like(z_vals),
        torch.zeros_like(z_vals),
    )
    # mask truncation and mask not hit voxel
    weights = weights * mask * sample_mask_per
    return weights / (torch.sum(weights, dim=-1, keepdims=True) + 1e-8), z_min


def render_rays(
        rays_o,
        rays_d,
        map_states,
        sdf_network,
        step_size,
        voxel_size,
        truncation,
        max_voxel_hit,
        max_distance,
        chunk_size=-1,
        profiler=None,
        return_raw=False,
        eval=False
):
    centres = map_states["voxel_center_xyz"]
    childrens = map_states["voxel_structure"]

    if profiler is not None:
        profiler.tick("ray_intersect")
    """
    intersections (dict):min_depth:(1,N_rays,N_hit_voxels) depth in camera coordinate if ray intersect voxel
            max_depth: (1,N_rays,N_hit_voxels)
            intersected_voxel_id: (1,N_rays,N_hit_voxels)
    hits:(1,N_rays),Whether each ray hits a voxel
    """
    intersections, hits = ray_intersect(
        rays_o, rays_d, centres,
        childrens, voxel_size, max_voxel_hit, max_distance)
    if profiler is not None:
        profiler.tok("ray_intersect")
    if hits.sum() == 0 and eval == True:
        ray_mask = torch.zeros_like(hits).bool().cuda()
        rgb = torch.zeros_like(rays_o).squeeze(0).cuda()
        depth = torch.zeros((rays_o.shape[1],)).cuda()
        return {
            "weights": None,
            "color": None,
            "depth": None,
            "z_vals": None,
            "sdf": None,
            "ray_mask": ray_mask,
            "raw": None if return_raw else None
        }

    else:
        assert (hits.sum() > 0)

    ray_mask = hits.view(1, -1)  # Whether each ray hits a voxel
    intersections = {
        name: outs[ray_mask].reshape(-1, outs.size(-1))
        for name, outs in intersections.items()
    }  # min_depth max_depth intersected_voxel_id: (N_rays,N_hit_voxels)

    rays_o = rays_o[ray_mask].reshape(-1, 3)  # remove rays which don't hit voxel
    rays_d = rays_d[ray_mask].reshape(-1, 3)

    """
    samples = {
        "sampled_point_depth": sampled_depth:(N_rays, N_points)
        "sampled_point_distance": sampled_dists:(N_rays, N_points)
        "sampled_point_voxel_idx": sampled_idx:(N_rays, N_points)
    }
    """
    samples = ray_sample(intersections, step_size=step_size)

    sampled_depth = samples['sampled_point_depth']
    sampled_idx = samples['sampled_point_voxel_idx'].long()

    # only compute when the ray hits, if don't hit setting False
    sample_mask = sampled_idx.ne(-1)
    if sample_mask.sum() == 0:  # miss everything skip
        return None, 0

    sampled_xyz = ray(rays_o.unsqueeze(
        1), rays_d.unsqueeze(1), sampled_depth.unsqueeze(2))
    samples['sampled_point_xyz'] = sampled_xyz

    # apply mask(remove don't hit the voxel)
    samples_valid = {name: s[sample_mask] for name, s in samples.items()}  # flatten to points

    num_points = samples_valid['sampled_point_depth'].shape[0]
    field_outputs = []
    if chunk_size < 0:
        chunk_size = num_points

    for i in range(0, num_points, chunk_size):
        chunk_samples = {name: s[i:i + chunk_size]
                         for name, s in samples_valid.items()}

        # get encoder features as inputs
        chunk_inputs = get_features(chunk_samples, map_states, voxel_size)

        # forward implicit fields
        if profiler is not None:
            profiler.tick("render_core")
        chunk_outputs = sdf_network(chunk_samples['sampled_point_xyz'])
        chunk_outputs['sdf'] = chunk_outputs['sdf'] + chunk_inputs['sdf_priors'][:, -1]

        field_outputs.append(chunk_outputs)

    field_outputs = {name: torch.cat(
        [r[name] for r in field_outputs], dim=0) for name in field_outputs[0]}

    outputs = {'sample_mask': sample_mask}

    sdf = masked_scatter_ones(sample_mask, field_outputs['sdf']).squeeze(-1)
    color = masked_scatter(sample_mask, field_outputs['color'])
    sample_mask = outputs['sample_mask']

    z_vals = samples["sampled_point_depth"]  # the depth from cam

    weights, z_min = sdf2weights(sdf, truncation, z_vals, sample_mask)

    rgb = torch.sum(weights[..., None] * color, dim=-2)

    depth = torch.sum(weights * z_vals, dim=-1)

    return {
        "weights": weights,
        "color": rgb,
        "depth": depth,
        "z_vals": z_vals,
        "sdf": sdf,
        "ray_mask": ray_mask,
        "raw": z_min if return_raw else None
    }


def bundle_adjust_frames(
        keyframe_graph,
        map_states,
        sdf_network,
        loss_criteria,
        voxel_size,
        step_size,
        N_rays=512,
        num_iterations=10,
        truncation=0.1,
        max_voxel_hit=10,
        max_distance=10,
        update_pose=True,
        batch_size=1024,
        optim=None,
        scaler=None,
        frame_id=None,
        use_adaptive_ending=False
):
    # sample rays from keyframes
    rays_o_all, rays_d_all, rgb_samples_all, depth_samples_all = [], [], [], []
    num_keyframe = len(keyframe_graph)
    for i, frame in enumerate(keyframe_graph):
        if update_pose:
            d_pose = frame.get_d_pose().cuda()
            ref_pose = frame.get_ref_pose().cuda()
            pose = d_pose @ ref_pose
        else:
            pose = frame.get_ref_pose().cuda()
        valid_idx = torch.nonzero(frame.valid_mask.reshape(-1))
        sample_idx = valid_idx[torch.randint(low=0, high=int(valid_idx.shape[0]),
                                             size=(int(num_iterations * (N_rays / num_keyframe)),))][:, 0]
        sampled_rays_d = frame.rays_d.cuda().reshape(-1, 3)[sample_idx]
        R = pose[: 3, : 3].transpose(-1, -2)
        sampled_rays_d = sampled_rays_d @ R
        sampled_rays_o = pose[: 3, 3].reshape(
            1, -1).expand_as(sampled_rays_d)
        rays_d_all += [sampled_rays_d]
        rays_o_all += [sampled_rays_o]
        rgb_samples_all += [frame.rgb.cuda().reshape(-1, 3)[sample_idx]]
        depth_samples_all += [frame.depth.cuda().reshape(-1)[sample_idx]]

    rays_d_all = torch.cat(rays_d_all, dim=0).unsqueeze(0)
    rays_o_all = torch.cat(rays_o_all, dim=0).unsqueeze(0)
    rgb_samples_all = torch.cat(rgb_samples_all, dim=0).unsqueeze(0)
    depth_samples_all = torch.cat(depth_samples_all, dim=0).unsqueeze(0)

    # shuffle
    shuffle_idx = torch.randperm(rays_d_all.shape[1])
    rays_d_all = rays_d_all[:, shuffle_idx].view(rays_d_all.size())
    rays_o_all = rays_o_all[:, shuffle_idx].view(rays_o_all.size())
    rgb_samples_all = rgb_samples_all[:, shuffle_idx].view(rgb_samples_all.size())
    depth_samples_all = depth_samples_all[:, shuffle_idx].view(depth_samples_all.size())

    loss_all = 0
    exceed_cnt = 0

    centres = map_states["voxel_center_xyz"]
    childrens = map_states["voxel_structure"]
    intersections, hits = ray_intersect(
        rays_o_all, rays_d_all, centres,
        childrens, voxel_size, max_voxel_hit, max_distance)
    assert (hits.sum() > 0)
    ray_mask = hits.view(1, -1)  # (1,N_rays),Whether each ray hits a voxel
    intersections = {
        name: outs.reshape(-1, outs.size(-1))
        for name, outs in intersections.items()
    }
    samples = ray_sample(intersections, step_size=step_size)
    sampled_depth = samples['sampled_point_depth']  # (N_rays,N_points)
    sampled_idx = samples['sampled_point_voxel_idx'].long()

    # only compute when the ray hits, if don't hit setting False
    sample_mask = sampled_idx.ne(
        -1)
    if sample_mask.sum() == 0:  # miss everything skip
        return None, 0

    sampled_xyz = ray(rays_o_all[0].unsqueeze(
        1), rays_d_all[0].unsqueeze(1), sampled_depth.unsqueeze(2))  # (N_rays,N_points,3)
    samples['sampled_point_xyz'] = sampled_xyz
    chunk_size = 100000000
    idx = 0
    for epoch in range(num_iterations):
        optim.zero_grad()
        # apply mask(remove don't hit the voxel)
        ray_mask_per = ray_mask[:, epoch * N_rays: (epoch + 1) * N_rays]
        sample_mask_per = sample_mask[epoch * N_rays: (epoch + 1) * N_rays][ray_mask_per[0]]
        samples_valid = {name: s[epoch * N_rays: (epoch + 1) * N_rays][ray_mask_per[0]][sample_mask_per] for name, s in
                         samples.items()}  # flatten to points
        num_points = samples_valid['sampled_point_depth'].shape[0]
        field_outputs = []
        with torch.cuda.amp.autocast():
            for i in range(0, num_points, chunk_size):
                chunk_samples = {name: s[i:i + chunk_size]
                                 for name, s in samples_valid.items()}
                chunk_inputs = get_features(chunk_samples, map_states, voxel_size)  # code in this
                chunk_outputs = sdf_network(chunk_samples['sampled_point_xyz'])
                chunk_outputs['sdf'] = chunk_outputs['sdf'] + chunk_inputs['sdf_priors'][:, -1]
                field_outputs.append(chunk_outputs)
            field_outputs = {name: torch.cat(
                [r[name] for r in field_outputs], dim=0) for name in
                field_outputs[0]}

            outputs = {'sample_mask': sample_mask_per}

            sdf = masked_scatter_ones(sample_mask_per, field_outputs['sdf']).squeeze(-1)
            color = masked_scatter(sample_mask_per, field_outputs['color'])
            sample_mask_per = outputs['sample_mask']

            z_vals = samples["sampled_point_depth"][epoch * N_rays: (epoch + 1) * N_rays][
                ray_mask_per[0]]  # (N_rays, N_samples_every_ray), the depth from cam
            weights, z_min = sdf2weights(sdf, truncation, z_vals, sample_mask_per)

            rgb = torch.sum(weights[..., None] * color, dim=-2)
            depth = torch.sum(weights * z_vals, dim=-1)
            final_outputs = {
                "weights": weights,
                "color": rgb,
                "depth": depth,
                "z_vals": z_vals,
                "sdf": sdf,
                "ray_mask": ray_mask_per,
            }
            loss = loss_criteria(
                final_outputs, (rgb_samples_all[:, epoch * N_rays: (epoch + 1) * N_rays].clone(),
                                depth_samples_all[:, epoch * N_rays: (epoch + 1) * N_rays].clone()))

        if use_adaptive_ending:
            loss_all += loss
            loss_mean = loss_all / (epoch + 1)
            if loss_mean < loss:
                exceed_cnt += 1
            else:
                exceed_cnt = 0
            if exceed_cnt >= 2 and frame_id != 0:
                break
        idx += 1
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
