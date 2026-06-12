import os

from dataset import DataLoaderTrain, DataLoaderReproduce, DataLoaderVal, DataLoaderTrainCPU, DataLoaderValCPU
def get_training_data(rgb_dir, num_training_samples):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, num_training_samples)

def get_reproduce_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderReproduce(rgb_dir)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir)

def get_training_dataCPU(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainCPU(rgb_dir)

def get_validation_dataCPU(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderValCPU(rgb_dir)