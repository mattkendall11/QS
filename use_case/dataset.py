# adapted from https://github.com/matteoprata/LOBCAST

import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils import data

from . import constants as cst

np.set_printoptions(suppress=True)


class FIDataset(data.Dataset):
    def __init__(
        self,
        dataset_type: str,
        dataset_path: str | Path,
        observation_length=cst.OBSERVATION_PERIOD,
    ):
        if dataset_type not in cst.DATASET_TYPES:
            raise ValueError("dataset_type must be in {}".format(cst.DATASET_TYPES))
        self.fi_data_dir = (
            Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        )
        self.dataset_type = dataset_type
        self.num_classes = cst.N_TRENDS
        self.horizons = cst.PREDICTION_HORIZON_FUTURE
        self.observation_length = observation_length

        # KEY call, generates the cst.Dataset
        self.data, self.samples_X, self.samples_y = None, None, None
        self.__prepare_dataset()

        # class balancing
        loss_weights = {}
        for i, horizon in enumerate(self.horizons):
            _, occs = self.__class_balancing(self.samples_y[:, i])
            LOSS_WEIGHT = 1e6
            loss_weights[horizon] = torch.Tensor(LOSS_WEIGHT / occs)

        self.samples_X = torch.from_numpy(self.samples_X).type(torch.FloatTensor)
        self.samples_y = torch.from_numpy(self.samples_y).type(torch.LongTensor)
        self.x_shape = (
            self.observation_length,
            self.samples_X.shape[1],
        )
        self.loss_weights = loss_weights

    def __len__(self):
        """Denotes the total number of samples."""
        return self.samples_X.shape[0] - self.observation_length

    def __getitem__(self, index):
        """Generates samples of data."""
        sample = (
            self.samples_X[index : index + self.observation_length],
            self.samples_y[index + self.observation_length - 1],
        )
        return sample

    @staticmethod
    def __class_balancing(y):
        ys_occurrences = collections.Counter(y)
        occs = np.array([ys_occurrences[k] for k in sorted(ys_occurrences)])
        return ys_occurrences, occs

    def __load_dataset(self):
        """Reads the dataset from the FI files."""
        self.data = np.load(self.fi_data_dir / "data_{}.npy".format(self.dataset_type))

    def __prepare_X(self):
        """we only consider the first 40 features, i.e. the 10 levels of the LOB"""
        LOB_TEN_LEVEL_FEATURES = 40
        self.samples_X = self.data[:LOB_TEN_LEVEL_FEATURES, :].transpose()
        self.samples_X = np.array(self.samples_X)

    def __prepare_y(self):
        """gets the labels"""
        samples_y = [
            self.data[cst.HORIZONS_MAPPINGS_FI[horizon], :] for horizon in self.horizons
        ]
        samples_y = np.array(samples_y).transpose()
        samples_y = samples_y - 1
        self.samples_y = samples_y

    def __prepare_dataset(self):
        """Crucial call!"""

        self.__load_dataset()
        self.__prepare_X()
        self.__prepare_y()
#data = Big_data('../data/BenchmarkDatasets', dataset_type='train', horizon=5, observation_length=10, train_val_split=0.8, n_trends=3)