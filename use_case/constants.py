# adapted from https://github.com/matteoprata/LOBCAST

from enum import Enum

HORIZONS_MAPPINGS_FI = {1: -5, 2: -4, 3: -3, 5: -2, 10: -1}

DATASET_TYPES = ["train", "test", "val"]
OBSERVATION_PERIOD = 10
PREDICTION_HORIZON_FUTURE = [1, 2, 3, 5, 10]
TRAIN_SET_PORTION = 0.8
N_TRENDS = 3


class FI_Horizons(Enum):
    K1 = 1
    K2 = 2
    K3 = 3
    K5 = 5
    K10 = 10


class Predictions(Enum):
    DOWNWARD = 0
    STATIONARY = 1
    UPWARD = 2


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"


class NormalizationType:
    Z_SCORE = 0
    DYNAMIC = 1
    NONE = 2
    MINMAX = 3
    DECPRE = 4


class Metrics(Enum):
    CM = "cm"
    F1 = "f1"
    F1_W = "f1_w"

    PRECISION = "precision"
    PRECISION_W = "precision_w"

    RECALL = "recall"
    RECALL_W = "recall_w"

    ACCURACY = "accuracy"
    MCC = "mcc"
    COK = "cohen-k"
