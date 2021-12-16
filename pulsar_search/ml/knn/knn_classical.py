from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from dataclasses  import dataclass

@dataclass
class DataSet:
    npts: int
    features: np.array
    labels: np.array

def read_dataset(dataset = '../data/htru2/HTRU_2.csv', delimiter=','):
    """Read the data set

    Args:
        dataset (str, optional): dataset filename. Defaults to '../data/htru2/HTRU_2.csv'.
        delimiter (str, optional): delimiter to read dataset. Defaults to ','.

    Returns:
        DataSet: Dataclass containing the dataset
    """

    data = np.loadtxt(dataset, delimiter=delimiter)
    dataset = DataSet(npts = len(data), features= data[:,:-1], labels= data[:,-1])
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

class KNN:

    def __init__(self, train_dataset, test_dataset, n_neighbor=3):
        """Handles the knn training/testing

        Args:
            train_dataset (dataclass): train dataset
            test_dataset (dataclass): test dataset
            n_neighbor (int, optional): number of neighbors. Defaults to 3.
        """

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_neighbor = n_neighbor
        self.model =  KNeighborsClassifier(n_neighbors=self.n_neighbor)

    def fit(self):
        """Fit the model
        """
        self.model.fit(self.train_dataset.features, self.train_dataset.labels)

    def test(self):
        """Test the model
        """"
        predict = self.model.predict(self.test_dataset.features)
        percent = np.sum(predict == self.test_dataset.labels)/self.test_dataset.npts
        print(percent)




if __name__ == "__main__":

    dataset = read_dataset()
    train_dataset, test_dataset = divide_dataset(dataset)

    knn = KNN(train_dataset, test_dataset)
    knn.fit()
    knn.test()