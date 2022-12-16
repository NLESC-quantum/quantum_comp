import numpy as np
from dataclasses  import dataclass



@dataclass
class DataSet:
    npts: int
    features: np.array
    labels: np.array

def read_dataset(dataset = 'frb_dataset.csv', delimiter=',', shuffle=False):
    """Read the data set

    Args:
        dataset (str, optional): dataset filename. Defaults to '../data/htru2/HTRU_2.csv'.
        delimiter (str, optional): delimiter to read dataset. Defaults to ','.

    Returns:
        DataSet: Dataclass containing the dataset
    """

    data = np.loadtxt(dataset, delimiter=delimiter)
    npts = len(data)
    idx = np.arange(npts)
    if shuffle:
        np.random.shuffle(idx)
    dataset = DataSet(npts = npts, features= data[idx,:-1], labels= data[idx,-1])
    return dataset


def balance_dataset(dataset, npts):

    idx0 = np.argwhere(dataset.labels==0).flatten()
    idx1 = np.argwhere(dataset.labels==1).flatten()
    idx0 = idx0[:idx1.size]
    idx = np.ravel(np.column_stack((idx0,idx1)))[:npts]

    dataset.npts = npts
    dataset.features = dataset.features[idx]
    dataset.labels = dataset.labels[idx]
    return dataset


def divide_dataset(dataset, fraction=[0.8,0.2], shuffle=True):
    """Divide the dataset in train and test dataset

    Args:
        dataset (Daatclass): dataset
        fraction (list, optional): how to divide train/test sets. Defaults to [0.8,0.2].
        shuffle (bool, optional): randomly change index of the dataset. Defaults to True.

    Returns:
        tuple: train and test datasets
    """
    index = np.arange(dataset.npts)

    if shuffle:
        np.random.shuffle(index)

    n_train  = int(fraction[0]*dataset.npts)
    n_test = dataset.npts-n_train
    idx_train = index[:n_train]
    idx_test = index[n_train:]
    train_dataset = DataSet(npts=n_train, features=dataset.features[idx_train,:], labels=dataset.labels[idx_train])
    test_dataset = DataSet(npts=n_test, features=dataset.features[idx_test,:], labels=dataset.labels[idx_test])

    return train_dataset, test_dataset