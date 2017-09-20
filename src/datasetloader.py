import torch.utils.data.dataloader as dataloader
import torch as torch
import csv
import numpy as np
import random
from dataset import ImdbDataset

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
        data = []
        with open(data_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            values = list(reader)
            data = np.array(values)
            header = data[0:1]
            data = data[1:data.shape[0]]
            random.shuffle(data)

        self.prepare_str_conplumn(data, 3)
        self.prepare_str_conplumn(data, 4)
        self.prepare_str_conplumn(data, 5)
        features = data[:, 0:27].astype(np.float32)
        images_urls = data[:, 27:28]
        self.normilize_columns(features, 2)
        self.normilize_columns(features, 3)
        self.normilize_columns(features, 4)
        self.normilize_columns(features, 5)
        return features[1 + round(part_start * len(features)):round(part_end * len(features))], images_urls

    def prepare_str_conplumn(self, data, column_index):
        uniq_values = np.unique(data[:, column_index])
        for i, lang in enumerate(uniq_values):
            data[data[:, column_index] == lang, column_index] = i

    def normilize_columns(self, data, column_index):
        data[:, column_index] = (data[:, column_index] - np.amin(data[:,column_index]))/ data[:,column_index].ptp(0)