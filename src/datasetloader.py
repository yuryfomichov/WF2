import torch.utils.data.dataloader as dataloader
import torch as torch
import csv
from .dataset import ImdbDataset

class DatasetLoader(object):
    def __init__(self, params):
        self.data_dir = 'dataset'
        self.train_file = 'train'
        self.test_file = 'test'
        self.image_folder = 'posters'
        self.batch_size = params.get("batch_size", 200)
        self.num_workers = params.get("num_workers", 32)

    def _get_loader(self,data, drop_last = False):
        imageFolder = '%s/%s/' % (self.data_dir, self.image_folder)
        loader = dataloader.DataLoader(ImdbDataset(data, imageFolder),
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       drop_last=drop_last,
                                       pin_memory=torch.cuda.is_available())
        return loader

    def get_train_loader(self, drop_last = True):
        return self._get_loader(self.get_data(self.train_file, 0, 0.8), drop_last)

    def get_val_loader(self):
        return self._get_loader(self.get_data(self.train_file, 0.8, 1), False)

    def get_test_loader(self):
        return self._get_loader(self.get_data(self.self.test_file), False)

    def get_data(self, file_name, part_start=0.0, part_end=1.0):
        data_file = '%s/%s.csv' % (self.data_dir, file_name)
        with open(data_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data = list(reader)
            return data[1+round(part_start*len(data)):round(part_end*len(data))]