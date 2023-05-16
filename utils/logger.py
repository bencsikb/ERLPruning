import os
import argparse
import json
import torch
import numpy as np
from varname import nameof
from torch.utils.tensorboard import SummaryWriter


class BasicLogger():

    def __init__(self, log_dir, test_case):
        super().__init__()

        self.log_dir = os.path.join(log_dir, test_case)
        self.test_case = test_case
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)

    def log_settings(self, opt, settings_dict=None):
        path = os.path.join(self.log_dir, "settings.txt")
        with open(path, 'w') as file:
            json.dump(opt.__dict__, file, indent=2)
            file.write("\n")
            file.write(json.dumps(settings_dict))

    def log_model(self, model):
        path = os.path.join(self.log_dir, "model.txt")
        with open(path, 'w') as file:
            file.write(str(model))


