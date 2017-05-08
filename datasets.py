import sys
import os
import time
import numpy as np

# courtesy of f0k:
# https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
def load_mnist():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    def label_to_one_hot(y_mat, num_classes=10):
        y_onehot = np.zeros((y_mat.shape[0], num_classes), dtype=y_mat.dtype)
        for i in range(y_mat.shape[0]):
          y_onehot[i][ y_mat[i] ] = 1.
        return y_onehot

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz').astype("float32")
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz').astype("float32")

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000].astype("float32"), X_train[-10000:].astype("float32")
    y_train, y_val = y_train[:-10000].astype("float32"), y_train[-10000:].astype("float32")

    y_train = label_to_one_hot(y_train)
    y_val = label_to_one_hot(y_val)
    y_test = label_to_one_hot(y_test)

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def iterator(X_full, y_full, bs, shuffle):
    b = 0
    if shuffle:
        idxs = [x for x in range(X_full.shape[0])]
        np.random.shuffle(idxs)
        X_full, y_full = X_full[idxs], y_full[idxs]
    while True:
        if b*bs >= X_full.shape[0]:
            break
        yield X_full[b*bs:(b+1)*bs], y_full[b*bs:(b+1)*bs]
        b += 1
