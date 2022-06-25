'''Feature set sensitivity test under the setting |S| = 2'''

import speck as sp

import numpy as np
from keras.models import model_from_json, load_model
from os import urandom
import copy
import random

block_size = 32


def make_masked_data(X, i1, i2):
    n = X.shape[0]
    masks_0 = np.frombuffer(urandom(n), dtype=np.uint8) & 0x1
    masks_1 = np.frombuffer(urandom(n), dtype=np.uint8) & 0x1
    X_with_same_masks = copy.deepcopy(X)
    X_with_random_masks = copy.deepcopy(X)

    X_with_same_masks[:, block_size - 1 - i1] = X[:, block_size - 1 - i1] ^ masks_0
    X_with_same_masks[:, block_size - 1 - i2] = X[:, block_size - 1 - i2] ^ masks_0

    X_with_random_masks[:, block_size - 1 - i1] = X[:, block_size - 1 - i1] ^ masks_0
    X_with_random_masks[:, block_size - 1 - i2] = X[:, block_size - 1 - i2] ^ masks_1

    return X_with_same_masks, X_with_random_masks


def test_FSS(n=10**7, nr=7, net_path='./', diff=(0x0040, 0x0), folder='./FSST_res/'):
    acc = np.zeros((block_size, block_size), dtype=np.float)
    X, Y = sp.make_train_data(n=n, nr=nr, diff=diff)
    if nr != 8:
        net = load_model(net_path)
    else:
        json_file = open('./saved_model/teacher/0x0040-0x0/single_block_resnet.json', 'r')
        json_model = json_file.read()
        net = model_from_json(json_model)
        net.load_weights(net_path)
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    loss, a0 = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is ', a0)

    for i1 in range(block_size):
        for i2 in range(i1 + 1, block_size):
            X_with_same_masks, X_with_random_masks = make_masked_data(X, i1, i2)
            loss, a1 = net.evaluate(X_with_random_masks, Y, batch_size=10000, verbose=0)
            loss, a2 = net.evaluate(X_with_same_masks, Y, batch_size=10000, verbose=0)
            acc[i1][i2] = a2 - a1
            print('cur bit connection is ', (i1, i2))
            print('the decrease of the acc is ', acc[i1][i2])

    np.save(folder + str(nr) + '_ND_bit_connection_sensitivity.npy', acc)


nr = 7
net_path = './saved_model/teacher/0x0040-0x0/{}_distinguisher.h5'.format(nr)
folder = './FSST_res/0x0040-0x0/'
test_FSS(n=10**6, nr=nr, net_path=net_path, diff=(0x0040, 0x0), folder=folder)

nr = 6
net_path = './saved_model/teacher/0x0040-0x0/{}_distinguisher.h5'.format(nr)
folder = './FSST_res/0x0040-0x0/'
test_FSS(n=10**6, nr=nr, net_path=net_path, diff=(0x0040, 0x0), folder=folder)