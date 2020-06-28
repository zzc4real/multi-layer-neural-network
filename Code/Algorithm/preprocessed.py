import numpy as np
import h5py
import sys


def train_vali_test_split(x, y, train_r, v_r, test_r):
    # num is the # of data samples
    num = x.shape[1]
    # total dataset will be split by two split points
    split1 = int(round(num * train_r))
    split2 = int(split1 + round(num * v_r))

    train_x = x[:, :split1]
    train_y = y[:, :split1]
    if (train_r == 0):
        train_x = None

    validation_x = x[:, split1:split2]
    validation_y = y[:, split1:split2]
    if (v_r == 0):
        validation_x = None

    test_x = x[:, split2:]
    test_y = y[:, split2:]
    if (test_r == 0):
        test_x = None

    return train_x, train_y, validation_x, validation_y, test_x, test_y



def data_normalization(data):
    mean = np.mean(data, axis= 1, keepdims = True)
    var = np.var(data, axis=1, keepdims=True)
    return (data - mean) / np.sqrt(var + 1e-10)


def data_preprocess(data_path):
    train_data = []
    label = []
    test_data = []

    # load data
    if (len(data_path)>0):
        with h5py.File(data_path + '/train_128.h5', 'r') as H:
            train_data = np.copy(H['data'])
        with h5py.File(data_path + '/train_label.h5', 'r') as H:
            label = np.copy(H['label'])
        with h5py.File(data_path + '/test_128.h5', 'r') as H:
            test_data = np.copy(H['data'])
    else:
        print("training data path is empty")
        sys.exit(0)

    # transfer the label to an one-hot form
    label = np.eye(np.max(label)+1)[label]

    # transpose matrix, a column is one train data
    train_data = train_data.T
    test_data = test_data.T
    label = label.T

    #  data normalization
    train_data = data_normalization(train_data)
    test_data = data_normalization(test_data)

    return train_data,test_data,label
