import numpy as np
import h5py
import cv2
import random
from keras.utils import np_utils

def load_mnist(path='/mnt/data/Dataset/MNIST/mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def load_usps(path='/mnt/data/Dataset/USPS/usps.h5'):
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        x_train = train.get('data')[:]
        y_train = train.get('target')[:]
        test = hf.get('test')
        x_test = test.get('data')[:]
        y_test = test.get('target')[:]

    return (x_train, y_train), (x_test, y_test)

def preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train = np.array([cv2.resize(img, (16, 16)) for img in x_train])
    x_test = np.array([cv2.resize(img, (16, 16)) for img in x_test])
    x_train = x_train.reshape(x_train.shape[0], 16, 16, 1)
    x_test = x_test.reshape(x_test.shape[0], 16, 16, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def preprocess_usps():
    (x_train, y_train), (x_test, y_test) = load_usps()
    x_train = x_train.reshape(x_train.shape[0], 16, 16, 1)
    x_test = x_test.reshape(x_test.shape[0], 16, 16, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    #y_train = np_utils.to_categorical(y_train, 10)
    #y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def sample_source_data(n_source_samples=2000):
    (x_train, y_train), (X_test, y_test) = preprocess_mnist()
    X_s = np.zeros((n_source_samples, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    y_s = np.zeros((n_source_samples, y_train.shape[1]))
    idxs = random.sample(range(len(x_train)), n_source_samples)
    for i, idx in enumerate(idxs):
        X_s[i] = x_train[idx]
        y_s[i] = y_train[idx]

    return X_s, y_s, X_test, y_test


def sample_target_data(n_target_samples=1):
    (x_train, y_train), (X_test, y_test) = preprocess_usps()
    X_t = np.zeros((n_target_samples * 10, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    y_t = np.zeros((n_target_samples * 10))
    classes = 10 * [n_target_samples]
    appended = []

    i = 0
    while True:
        if i == n_target_samples * 10:
            break
        idx = random.randint(0, x_train.shape[0])
        if (not idx in appended) and classes[y_train[idx]] > 0:
            X_t[i] = x_train[idx]
            y_t[i] = y_train[idx]
            classes[y_train[idx]] -= 1
            appended.append(idx)
            i += 1

    y_t = np_utils.to_categorical(y_t, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return X_t, y_t, X_test, y_test

def create_pairs(X_s, y_s, X_t, y_t):
    t_p = []
    t_n = []
    for i in range(y_s.shape[0]):
        for j in range(y_t.shape[0]):
            if (y_s[i] == y_t[j]).all():
                t_p.append((i, j))
            else:
                t_n.append((i, j))
    
    random.shuffle(t_n)
    t = t_p + t_n[: 3*len(t_p)]
    random.shuffle(t)

    X1 = np.zeros((len(t), X_s.shape[1], X_s.shape[2], X_s.shape[3]), dtype='float32')
    X2 = np.zeros((len(t), X_t.shape[1], X_t.shape[2], X_t.shape[3]), dtype='float32')

    y1 = np.zeros((len(t), y_s.shape[1]))
    y2 = np.zeros((len(t), y_t.shape[1]))
    yc = np.zeros(len(t))

    for i in range(len(t)):
        idx1, idx2 = t[i]
        X1[i] = X_s[idx1]
        X2[i] = X_t[idx2]
        y1[i] = y_s[idx1]
        y2[i] = y_t[idx2]
        if (y1[i] == y2[i]).all():
            yc[i] = 1

    yc = np_utils.to_categorical(yc, 10)

    return X1, X2, y1, y2, yc

if __name__ == '__main__':
    """
    X_s, y_s, x_test, y_test = sample_source_data()
    print(X_s.shape)
    print(x_test.shape)
    print(y_s.shape)
    print(y_test.shape)
    """
    X_s, y_s, x_test, y_test = sample_source_data()
    X_t, y_t, x_test, y_test = sample_target_data()
    X1, X2, y1, y2, yc = create_pairs(X_s, y_s, X_t, y_t)
    print(X1.shape)
    print(X2.shape)
    print(y1.shape)
    print(y1.shape)
    print(yc.shape)