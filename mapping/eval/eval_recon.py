import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, ".")  # noqa
sys.path.insert(0, os.path.abspath('src'))  # noqa
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree
from demo.parser import get_parser
from tqdm import tqdm


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(float))
    comp_ratio_std = np.std((distances < dist_th).astype(float))
    return comp_ratio, comp_ratio_std


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    acc_std = np.std(distances)
    return acc, acc_std


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    comp_std = np.std(distances)
    return comp, comp_std


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation = reg_p2p.transformation
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.
    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()  # (N, 4)
    cam_cord_homo = w2c @ homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float() @ cam_cord.float()
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
    return mask.sum() > 0


def calc_3d_metric(args, align=True):
    """
    3D reconstruction metric.
    """
    rec_meshfile = args.rec_mesh
    gt_meshfile = args.gt_mesh
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    only_cal_has_vox = args.only_cal_has_vox
    if only_cal_has_vox:
        offset = args.mapper_specs["offset"]
        ckpt = torch.load(args.ckpt)
        vox_cen_xyz = ckpt['map_state']['voxel_center_xyz']
        vox_vert_idx = ckpt['map_state']['voxel_vertex_idx']
        vox_cen_xyz = vox_cen_xyz[~vox_vert_idx.eq(-1).any(-1)] - offset

        batch_size = 1000
        vox_cen_xyz = vox_cen_xyz.to('cuda')
        mask_has_voxel = np.ones(mesh_gt.vertices.shape[0]).astype(bool)
        for batch_iter in tqdm(range(0, mesh_gt.vertices.shape[0], batch_size)):
            vertex = torch.tensor(mesh_gt.vertices[batch_iter:batch_iter + batch_size, :]).to('cuda')
            dis = vox_cen_xyz.unsqueeze(0) - vertex.unsqueeze(1)
            # tmp = (dis.abs() < args.mapper_specs["voxel_size"] / 2).all(dim=-1).any(dim=-1)
            tmp = (dis.abs() < (args.mapper_specs["voxel_size"] / 2 + 1e-5)).all(dim=-1).any(dim=-1)
            mask_has_voxel[batch_iter:batch_iter + batch_size] = tmp.cpu().numpy()

        face_mask = mask_has_voxel[mesh_gt.faces].all(axis=1)
        mesh_gt.update_faces(face_mask)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface_even(mesh_rec,
                                                200000)  # Sample the surface of a mesh, returning the specified number of points
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface_even(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec, accuracy_rec_std = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec, completion_rec_std = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec, completion_ratio_rec_std = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    accuracy_rec_std *= 100  # convert to cm
    completion_rec_std *= 100  # convert to cm
    completion_ratio_rec_std *= 100  # convert to %
    print('accuracy: ', accuracy_rec)
    print('completion: ', completion_rec)
    print('completion ratio: ', completion_ratio_rec)
    print('accuracy std: ', accuracy_rec_std)
    print('completion std: ', completion_rec_std)
    print('completion ratio std: ', completion_ratio_rec_std)
    np.savetxt(args.out_dir + "/3d_matrics.txt", np.array([accuracy_rec, completion_rec, completion_ratio_rec]))
    np.savetxt(args.out_dir + "/3d_matrics_std.txt",
               np.array([accuracy_rec_std, completion_rec_std, completion_ratio_rec_std]))


def get_cam_position(gt_meshfile):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= 0.7
    extents[1] *= 0.7
    extents[0] *= 0.3
    transform = np.linalg.inv(to_origin)
    transform[2, 3] += 0.4
    return extents, transform


def calc_2d_metric(args, align=True, n_imgs=1000):
    """
    2D reconstruction metric, depth L1 loss.
    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H / 2.0 - 0.5
    cy = W / 2.0 - 0.5

    rec_meshfile = args.rec_mesh
    gt_meshfile = args.gt_mesh
    rec_mesh = trimesh.load(rec_meshfile, process=False)
    gt_mesh = trimesh.load(gt_meshfile, process=False)

    only_cal_has_vox = args.only_cal_has_vox
    if only_cal_has_vox:
        offset = args.mapper_specs["offset"]
        ckpt = torch.load(args.ckpt)
        vox_cen_xyz = ckpt['map_state']['voxel_center_xyz']
        vox_vert_idx = ckpt['map_state']['voxel_vertex_idx']
        vox_cen_xyz = vox_cen_xyz[~vox_vert_idx.eq(-1).any(-1)] - offset

        batch_size = 1000
        vox_cen_xyz = vox_cen_xyz.to('cuda')
        mask_has_voxel = np.ones(gt_mesh.vertices.shape[0]).astype(bool)
        for batch_iter in tqdm(range(0, gt_mesh.vertices.shape[0], batch_size)):
            vertex = torch.tensor(gt_mesh.vertices[batch_iter:batch_iter + batch_size, :]).to('cuda')
            dis = vox_cen_xyz.unsqueeze(0) - vertex.unsqueeze(1)
            tmp = (dis.abs() < (args.mapper_specs["voxel_size"] / 2 + 1e-5)).all(dim=-1).any(dim=-1)
            mask_has_voxel[batch_iter:batch_iter + batch_size] = tmp.cpu().numpy()

        face_mask = mask_has_voxel[gt_mesh.faces].all(axis=1)
        gt_mesh.update_faces(face_mask)
    gt_mesh.export('gt_mesh.ply')
    rec_mesh.export('rec_mesh.ply')

    gt_mesh = o3d.io.read_triangle_mesh('gt_mesh.ply')
    rec_mesh = o3d.io.read_triangle_mesh('rec_mesh.ply')

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in tqdm(range(n_imgs)):
        while True:
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(
                extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            target = [tx, ty, tz]
            target = np.array(target) - np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w
            c2w = tmp

            param = o3d.camera.PinholeCameraParameters()
            param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

            param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                W, H, fx, fy, cx, cy)

            ctr = vis.get_view_control()
            ctr.set_constant_z_far(20)
            ctr.convert_from_pinhole_camera_parameters(param)

            vis.add_geometry(gt_mesh, reset_bounding_box=True, )
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            gt_depth = vis.capture_depth_float_buffer(True)
            gt_depth = np.asarray(gt_depth)
            vis.remove_geometry(gt_mesh, reset_bounding_box=True, )
            if (gt_depth != 0).any():
                break

        vis.add_geometry(rec_mesh, reset_bounding_box=True, )
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(True)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(rec_mesh, reset_bounding_box=True, )

        mask = (gt_depth != 0) * (ours_depth != 0)
        errors += [np.abs(gt_depth[mask] - ours_depth[mask]).mean()]

    errors = np.array(errors)
    # m
    print('Depth L1: ', errors.mean())
    np.savetxt(args.out_dir + "/mesh_2d_matrics_mean.txt", np.array([errors.mean()]))
    np.savetxt(args.out_dir + "/mesh_2d_matrics_std.txt", np.array([errors.std()]))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    setup_seed(12345)
    torch.classes.load_library(
        "third_party/sparse_octree/build/lib.linux-x86_64-cpython-38/svo.cpython-38-x86_64-linux-gnu.so")

    parser = get_parser()
    parser.add_argument('-only_cal_has_vox', action='store_true',
                        help='remove faces from a gt_mesh that are occluded in all RGB-D frames', default=True)
    parser.add_argument('--rec_mesh', type=str,
                        help='reconstructed mesh file path')
    parser.add_argument('--gt_mesh', type=str,
                        help='ground truth mesh file path')
    parser.add_argument('-2d', '--metric_2d',
                        action='store_true', help='enable 2D metric')
    parser.add_argument('-3d', '--metric_3d',
                        action='store_true', help='enable 3D metric')
    parser.add_argument('--ckpt', type=str,
                        help='ckpt file path')  # ckpt from our work
    parser.add_argument('--out_dir', type=str,
                        help='output file path')
    args = parser.parse_args()

    if args.metric_3d:
        calc_3d_metric(args)

    if args.metric_2d:
        calc_2d_metric(args, n_imgs=1000)
