import os
import argparse
import json
import torch
import numpy as np
from varname import nameof
from torch.utils.tensorboard import SummaryWriter
from utils.config_parser import ConfigParser


class BasicLogger():

    def __init__(self, log_dir, test_case):
        super().__init__()

        self.log_dir = os.path.join(log_dir, test_case)
        self.test_case = test_case
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)

    def log_settings(self, conf, settings_dict=None):
        # TODO this should be a method in ConfigPrser
        path = os.path.join(conf.paths.log_dir, conf.logging.folder, conf.dynamic.test_case, "settings.txt")
        ConfigParser.save(conf, path)
        """
        with open(path, 'w') as file:
            json.dump(opt.__dict__, file, indent=2)
            file.write("\n")
            file.write(json.dumps(settings_dict))
        """

    def log_model(self, model):
        path = os.path.join(self.log_dir, "model.txt")
        with open(path, 'w') as file:
            file.write(str(model))


