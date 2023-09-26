import os
import sys

sys.path.insert(0, os.path.abspath('src'))
import random
from parser import get_parser
import numpy as np
import torch
from H2Mapping import H2Mapping


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = get_parser().parse_args()
    if hasattr(args, 'seeding'):
        setup_seed(args.seeding)
    else:
        setup_seed(12345)
    slam = H2Mapping(args)
    slam.run()
