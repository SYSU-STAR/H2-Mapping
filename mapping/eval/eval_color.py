import os
import sys

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, ".")  # noqa
sys.path.insert(0, os.path.abspath('src'))  # noqa
from demo.parser import get_parser
from src.utils.import_util import get_dataset
from src.utils.import_util import get_decoder
import sys
import os
from src.frame import RGBDFrame
from src.loggers import BasicLogger
from src.mapping import Mapping
import numpy as np
from tqdm import tqdm
from record_video import make_video

sys.path.insert(0, os.path.abspath('src'))  # noqa

if __name__ == '__main__':
    torch.classes.load_library(
        "third_party/sparse_octree/build/lib.linux-x86_64-cpython-38/svo.cpython-38-x86_64-linux-gnu.so")

    parser = get_parser()
    parser.add_argument('--result_file', type=str, help='output result file path')
    args = parser.parse_args()

    result_file = args.result_file

    ckpt_path = result_file + "/ckpt/final_ckpt.pth"
    training_result = torch.load(ckpt_path)

    decoder = get_decoder(args).cuda()

    data_stream = get_dataset(args)
    data_in = data_stream[0]
    first_frame = RGBDFrame(*data_in[:-1], offset=args.mapper_specs['offset'], ref_pose=data_in[-1]).cuda()
    W, H = first_frame.rgb.shape[1], first_frame.rgb.shape[0]

    args.debug_args['render_res'] = [W, H]  # image size

    logger = BasicLogger(args, for_eva=True)
    logger.img_dir_rgb_render = result_file + "/mapping_vis/color"
    logger.img_dir_depth_render = result_file + "/mapping_vis/depth"
    if not os.path.exists(logger.img_dir_rgb_render):
        os.makedirs(logger.img_dir_rgb_render)
    if not os.path.exists(logger.img_dir_depth_render):
        os.makedirs(logger.img_dir_depth_render)
    mapper = Mapping(args, logger, data_stream=data_stream)
    mapper.decoder.load_state_dict(training_result['decoder_state'])
    mapper.sdf_priors = training_result['sdf_priors'].cuda()
    mapper.map_states = training_result['map_state']
    mapper.decoder = mapper.decoder.cuda()
    mapper.decoder.eval()

    ssim_all, psnr_all, depth_L1_err_all = [], [], []
    for i in tqdm(range(len(data_stream))):
        data_in = data_stream[i]
        tracked_frame = RGBDFrame(*data_in[:-1], offset=args.mapper_specs['offset'], ref_pose=data_in[-1]).cuda()
        W, H = tracked_frame.rgb.shape[1], tracked_frame.rgb.shape[0]
        args.debug_args['render_res'] = [W, H]  # image size
        if tracked_frame.ref_pose.isinf().any():
            continue
        ssim, psnr, depth_L1_err = mapper.render_debug_images(tracked_frame)
        ssim_all.append(ssim)
        psnr_all.append(psnr)
        depth_L1_err_all.append(depth_L1_err)

    ssim_all = np.array(ssim_all)
    psnr_all = np.array(psnr_all)
    depth_L1_err_all = np.array(depth_L1_err_all)
    print('ssim mean: ', ssim_all.mean(), 'ssim std: ', ssim_all.std())
    print('psnr mean: ', psnr_all.mean(), 'psnr std: ', psnr_all.std())
    print('depth_L1 mean: ', depth_L1_err_all.mean(), 'depth_L1 std: ', depth_L1_err_all.std())
    np.savetxt(
        result_file + "/ssim_mean" + str(ssim_all.mean()) + '_std_' + str(ssim_all.std()) + ".txt", ssim_all)
    np.savetxt(
        result_file + "/psnr_mean" + str(psnr_all.mean()) + '_std_' + str(psnr_all.std()) + ".txt", psnr_all)
    np.savetxt(result_file + "/depth_L1_mean" + str(depth_L1_err_all.mean()) + '_std_' + str(
        depth_L1_err_all.std()) + ".txt", depth_L1_err_all)
    mean = np.array([ssim_all.mean(), psnr_all.mean(), depth_L1_err_all.mean()])
    std = np.array([ssim_all.std(), psnr_all.std(), depth_L1_err_all.std()])
    np.savetxt(result_file + "/color_psnr_ssim_depth_L1_mean.txt", mean)
    np.savetxt(result_file + "/color_psnr_ssim_depth_L1_std.txt", std)

    make_video(file_path=result_file, output_path=result_file, fps=30, size=(W, H), total_frame=len(data_stream))
