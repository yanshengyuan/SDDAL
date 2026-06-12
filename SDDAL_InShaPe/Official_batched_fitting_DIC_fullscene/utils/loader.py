import os

from dataset import DataLoaderTrain # For fitting script, use this
#from Official_batched_fitting_DIC_fullscene.dataset import DataLoaderTrain # For the sanity check of Scanner.py and Initializer.py, use this

def get_training_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir)