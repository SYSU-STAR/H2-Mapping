import numpy as np
import torch
import sys
import os
sys.path.insert(0, ".") # noqa
sys.path.insert(0, os.path.abspath('src')) # noqa
from demo.parser import get_parser
from utils.import_util import get_dataset
from utils.import_util import get_decoder, get_property
import sys
import os
import os.path as osp
from frame import RGBDFrame
from loggers import BasicLogger
from mapping import Mapping
import numpy as np
from tqdm import tqdm
from viz import SLAMFrontend
import time

if __name__ == '__main__':
    torch.classes.load_library(
        "third_party/sparse_octree/build/lib.linux-x86_64-cpython-38/svo.cpython-38-x86_64-linux-gnu.so")

    parser = get_parser(vis=True)
    parser.add_argument('--result_file', type=str, help='output result file path')
    parser.add_argument('-create_mesh', '--create_mesh',action='store_true')
    parser.add_argument('-save_rendering', '--save_rendering', action='store_true', default=False)
    args = parser.parse_args()

    result_file = args.result_file

    pose_path = result_file + "/misc/frame_poses.npy"
    pose_all = np.load(pose_path)
    ckpt_list = os.listdir(result_file + "/ckpt/")
    if args.create_mesh:
        for ckpt in ckpt_list:
            if ckpt == 'final_ckpt.pth':
                continue
            data_path = result_file + "/ckpt/" + ckpt
            training_result = torch.load(data_path)

            decoder = get_decoder(args).cuda()
            if args.run_ros:
                data_stream = None
            else:
                data_stream = get_dataset(args)

            logger = BasicLogger(args, for_eva=True)
            logger.mesh_dir = result_file + "/mesh"
            if not os.path.exists(logger.mesh_dir):
                os.makedirs(logger.mesh_dir)

            mapper = Mapping(args, logger, data_stream=data_stream)
            mapper.decoder.load_state_dict(training_result['decoder_state'])
            mapper.sdf_priors = training_result['sdf_priors'].cuda()
            mapper.map_states = training_result['map_state']
            mapper.decoder = mapper.decoder.cuda()
            mapper.decoder.eval()

            mesh = mapper.extract_mesh(res=args.mapper_specs['mesh_res'], clean_mesh=False,
                                       map_states=mapper.map_states)
            logger.log_mesh(mesh, name=f"mesh_{ckpt[:-4]}.ply")

    mesh_bound = np.array(args.decoder_specs['bound']) - np.array(args.mapper_specs['offset'])
    init_pose = np.eye(4)
    init_pose[:3, 3] = mesh_bound.mean(-1)  # move to the center

    # room0
    init_pose[1, 3] += 1
    init_pose[0, 3] -= 4
    init_pose[2, 3] += 8  # up

    # bird eye
    init_pose[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    fov = 0
    near = 0

    result_dir = result_file

    frontend = SLAMFrontend(result_dir=result_dir, init_pose=init_pose, cam_scale=0.3,
                            save_rendering=args.save_rendering, near=near,
                            estimate_c2w_list=pose_all, gt_c2w_list=pose_all, fov=fov).start()

    for i in tqdm(range(pose_all.shape[0])):
        print(i)
        time.sleep(0.03)
        meshfile = f'{result_dir}/mesh/mesh_{i:06d}.ply'
        if os.path.isfile(meshfile):
            frontend.update_mesh(meshfile)
        frontend.update_pose(1, pose_all[i], gt=False)
        # the visualizer might get stucked if update every frame
        # with a long sequence (10000+ frames)
        if i % 1 == 0:
            frontend.update_cam_trajectory(i, gt=False)