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
        self._load_data()

    def _get_loader(self,data, drop_last = False, is_train = True):
        imageFolder = '%s/%s/' % (self.data_dir, self.image_folder)
        loader = dataloader.DataLoader(ImdbDataset(data, imageFolder, is_train),
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       drop_last=drop_last,
                                       pin_memory=torch.cuda.is_available())
        return loader

    def get_train_loader(self, drop_last = True):
        return self._get_loader(self.train_data, drop_last, True)

    def get_val_loader(self):
        return self._get_loader(self.val_data, False, False)

    def get_test_loader(self):
        return self._get_loader(self.test_data, False, False)

    def _prepare_data(self, data):

        features = data[:, 0:27].astype(np.float32)
        images_urls = data[:, 27:28]
        self.normalize_columns(features, 2)
        self.normalize_columns(features, 3)
        self.normalize_columns(features, 4)
        self.normalize_columns(features, 5)


    def prepare_str_column(self, data, column_index, uniq_values):
        for i, lang in enumerate(uniq_values):
            data[data[:, column_index] == lang, column_index] = i

    def normalize_columns(self, data, column_index_list):
        for column_index in column_index_list:
            column = data[:,column_index].astype(np.float32);
            data[:, column_index] = ((column - np.amin(column))/column.ptp(0)).astype(np.str)

    def _load_data(self):
        train_file_data = self._read_csv(self.train_file)
        train_file_data = train_file_data[1:]
        uniq_countries = np.unique(train_file_data[:, 3])
        uniq_lang = np.unique(train_file_data[:, 4])
        uniq_rate = np.unique(train_file_data[:, 5])
        self.prepare_str_column(train_file_data, 3, uniq_countries)
        self.prepare_str_column(train_file_data, 4, uniq_lang)
        self.prepare_str_column(train_file_data, 5, uniq_rate)
        self.normalize_columns(train_file_data, [2,3,4,5])
        length = len(train_file_data)

        # self.test_data = self.slice_data(train_file_data, 0.92, 1, length)
        # self.train_data = train_file_data[0: round(0.92 * length)]
        # np.random.shuffle(self.train_data)
        # length = len(self.train_data)
        # self.val_data = self.slice_data(self.train_data, 0.9, 1, length)
        # self.train_data = self.slice_data(self.train_data, 0, 0.9, length)

        self.test_data = self.slice_data(train_file_data, 0.9, 1, length)
        self.val_data = self.slice_data(train_file_data, 0.9, 1, length)
        self.train_data = train_file_data[0: round(0.9 * length)]
        np.random.shuffle(self.train_data)
        self.train_data = self.slice_data(self.train_data, 0, 1, len(self.train_data))

        pass

    def slice_data(self, data, start, stop, length):
        data = data[round(start*length): round(stop*length)]
        features = data[:, 0:27].astype(np.float32)
        images_urls = data[:, 27:28]
        return features, images_urls

    def _read_csv(self, file_name):
        data_file = '%s/%s.csv' % (self.data_dir, file_name)
        with open(data_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            values = list(reader)
            data = np.array(values)
        return data