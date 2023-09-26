from loggers import BasicLogger
from mapping import Mapping
from utils.import_util import get_dataset


class H2Mapping:
    def __init__(self, args):
        self.args = args
        # logger (optional)
        self.logger = BasicLogger(args)
        if not args.run_ros:
            # data stream
            self.data_stream = get_dataset(args)
        else:
            self.data_stream = None
        # mapper
        self.mapper = Mapping(args, self.logger, data_stream=self.data_stream)
        # initialize map with first frame
        self.update_pose = args.update_pose
        if not args.run_ros:
            self.firstframe = self.mapper.initfirst_onlymap()
        else:
            self.firstframe = None

    def run(self):
        self.mapper.run(self.firstframe, update_pose=self.update_pose)
