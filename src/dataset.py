import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.io import loadmat
from os.path import join, dirname
from tensorflow.keras.utils import get_file


def download_existing_dataset(dataset_url, dataset_name):
    '''Download Dataset
    Params:
        dataset_url     -> Url to the dataset
        dataset_name    -> Name of the dataset
    '''
    file_name = dataset_url.split('/')[-1]
    path = get_file(
        file_name,
        origin=dataset_url,
        extract=True
    )
    return path


def load_data(dataset_path):
    '''Load Voxel Data
    Params:
        dataset_path -> Path to the dataset
    '''
    x = []
    mat_files = glob(join(dataset_path, '*/*.mat'))
    for _file in tqdm(mat_files):
        data = loadmat(_file)
        x.append(data['voxel'])
    return np.array(x).astype(np.float32)