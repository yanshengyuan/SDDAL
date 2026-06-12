import os

from dataset import DataLoaderTrain, DataLoaderVal
def get_training_data(rgb_dir, num_training_samples):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, num_training_samples)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir)
