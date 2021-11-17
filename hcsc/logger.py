import torch
import os

class EasyLogger(object):
    def __init__(self, save_dir="", epoch=0, rank=0):
        self.save_dir = save_dir
        self.epoch = epoch
        self.rank = rank

    def set_epoch(self, epoch):
        self.epoch = epoch

    def save_file(self, save_dict, save_name):
        torch.save(save_dict, os.path.join(self.save_dir, save_name+"_rank{}".format(self.rank)))

    def save_with_epoch(self, save_dict, save_name):
        torch.save(save_dict, os.path.join(self.save_dir, save_name+ "_rank{}".format(self.rank) + "_epoch{}".format(self.epoch)))