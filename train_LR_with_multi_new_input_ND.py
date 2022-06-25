import numpy as np
import speck as sp
from keras.models import load_model
from sklearn.linear_model import LogisticRegression as LR
from sklearn.externals import joblib
from os import urandom, path, mkdir
import time


def convert_to_binary_for_new_input(arr):
    X = np.zeros((3 * sp.WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(3 * sp.WORD_SIZE()):
        index = i // sp.WORD_SIZE()
        offset = sp.WORD_SIZE() - (i % sp.WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return(X)


def extract_sensitive_bits_for_new_input(raw_x, bits=[14,13,12,11,10,9,8,7]):
    # get new-x according to sensitive bits
    id0 = [15 - v for v in bits]
    id1 = [v + 16 * i for i in range(3) for v in id0]
    new_x = raw_x[:, id1]
    return new_x


# new input: (c0l ^ c1l, c0r ^ c1r, c0l ^ c0r)
def make_raw_data_with_new_input(n, nr, diff=(0x0040, 0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y == 0)
    plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    ks = sp.expand_key(keys, nr)
    ctdata0l, ctdata0r = sp.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = sp.encrypt((plain1l, plain1r), ks)

    t0, t1, t2 = ctdata0l ^ ctdata1l, ctdata0r ^ ctdata1r, ctdata0l ^ ctdata0r
    X = convert_to_binary_for_new_input([t0, t1, t2])
    return (X, Y)


def make_dataset_for_LR(n=10**7, nr=6, diff=(0x0040, 0), nets=None, bits=None):
    x, y = make_raw_data_with_new_input(n=n, nr=nr, diff=diff)
    new_x = np.zeros((n, len(nets)), dtype=np.float)

    num = len(nets)
    for i in range(num):
        nd = load_model(nets[i])
        extracted_x = extract_sensitive_bits_for_new_input(x, bits=bits[i])
        z = nd.predict(extracted_x, batch_size=10**4, verbose=0)
        new_x[:, i] = np.squeeze(z)
        print('the ', i, '-th nd finished')

    return new_x, y


def make_dataset_for_LR_with_teacher(n=10**7, nr=6, diff=(0x0040, 0), nets=None, bits=None, teacher=None):
    raw_x, raw_y = make_raw_data_with_new_input(n=n, nr=nr, diff=diff)
    new_x = np.zeros((n, len(nets)), dtype=np.float)
    # make new labels according to the teacher
    tnet = load_model(teacher)
    z = tnet.predict(raw_x, batch_size=10 ** 4, verbose=0)
    new_y = np.where(z > 0.5, 1, 0)
    # reshape into 1D vector
    new_y = new_y.ravel()

    num = len(nets)
    for i in range(num):
        nd = load_model(nets[i])
        extracted_x = extract_sensitive_bits_for_new_input(raw_x, bits=bits[i])
        z = nd.predict(extracted_x, batch_size=10 ** 4, verbose=0)
        new_x[:, i] = np.squeeze(z)
        print('the ', i, '-th nd finished')

    return new_x, new_y


def train_LR(nr=6, diff=(0x0040, 0), nets=None, bits=None, teacher=None):
    # make training data without teacher
    X, Y = make_dataset_for_LR(n=10**7, nr=nr, diff=diff, nets=nets, bits=bits)
    # # make training data with teacher
    # X, Y = make_dataset_for_LR_with_teacher(n=10 ** 7, nr=nr, diff=diff, nets=nets, bits=bits, teacher=teacher)
    # make testing data
    X_eval, Y_eval = make_dataset_for_LR(n=10**6, nr=nr, diff=diff, nets=nets, bits=bits)

    # build proxy model for feature selection and bit independence test
    proxy_net_1 = LR(C=0.01, penalty='l1', tol=0.00001, solver='saga', max_iter=100, verbose=1)
    proxy_net_1.fit(X, Y)
    acc1 = proxy_net_1.score(X_eval, Y_eval)
    print('the testing acc is ', acc1)
    # print(proxy_net.get_params(deep=True))
    print('coefs are ', proxy_net_1.coef_)
    print('intercept is ', proxy_net_1.intercept_)

# folder = './saved_model/new_inputs/'
# net_paths = [
#     folder + '0.7173_14_10_5_3_student_6_distinguisher.h5',      # 1
#     folder + '0.7206_12_9_4_1_student_6_distinguisher.h5'       # 4
# ]
# selected_bits = [
#     [14, 13, 12, 11, 10, 5, 4, 3],      # 1
#     [12, 11, 10, 9, 4, 3, 2, 1],        # 4
# ]
# teacher = './saved_model/teacher/0x0040-0x0/6_distinguisher.h5'
#
# # when the raw input is adopted:
# # when two NDs (1, 4) are used, and the training method is normal,
# # the testing acc is  0.738363
# # coefs are  [[2.85744491 3.62420957]]
# # intercept is  [-3.1249105]
#
# # when the new input is adopted:
# # when two NDs (1, 4) are used, and the training method is normal,
# # the testing acc is  0.738014
# # coefs are  [[2.82072909 3.64216749]]
# # intercept is  [-3.13645595]
#
# train_LR(nr=6, diff=(0x0040, 0), nets=net_paths, bits=selected_bits, teacher=teacher)


folder = './saved_model/new_inputs/'
net_paths = [
    folder + '0.5886_14_9_5_4_student_7_distinguisher.h5',       # 0
    folder + '0.538_12_11_5_1_student_7_distinguisher.h5'       # 5
]

selected_bits = [
    [14, 13, 12, 11, 10, 9, 5, 4],  # 0
    [12, 11, 5, 4, 3, 2, 1]      # 5
]

teacher = './saved_model/teacher/0x0040-0x0/7_distinguisher.h5'

# when the raw input is adopted:
# when two NDs (0, 5) are adopted, and the training method is normal,
# the testing acc is  0.595568
# coefs are  [[4.12507959 3.61160246]]
# intercept is  [-3.86370923]

# when the new input is adopted:
# when two NDs (0, 5) are adopted, and the training method is normal,
# the testing acc is  0.595492
# coefs are  [[4.08646331 3.4168975 ]]
# intercept is  [-3.7528234]

train_LR(nr=7, diff=(0x0040, 0), nets=net_paths, bits=selected_bits, teacher=teacher)