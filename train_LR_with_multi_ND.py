import numpy as np
import speck as sp
from keras.models import load_model
from sklearn.linear_model import LogisticRegression as LR
from sklearn.externals import joblib
from os import urandom, path, mkdir
import time


def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [15 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + 16 * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]

    return new_x


def make_dataset_for_LR(n=10**7, nr=6, diff=(0x0040, 0), nets=None, bits=None):
    raw_x, raw_y = sp.make_train_data(n=n, nr=nr, diff=diff)
    new_x = np.zeros((n, len(nets)), dtype=np.float)

    num = len(nets)
    for i in range(num):
        nd = load_model(nets[i])
        extracted_x = extract_sensitive_bits(raw_x, bits=bits[i])
        z = nd.predict(extracted_x, batch_size=10**4, verbose=0)
        new_x[:, i] = np.squeeze(z)
        print('the ', i, '-th nd finished')

    return new_x, raw_y


def make_dataset_for_LR_with_teacher(n=10**7, nr=6, diff=(0x0040, 0), nets=None, bits=None, teacher=None):
    raw_x, raw_y = sp.make_train_data(n=n, nr=nr, diff=diff)
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
        extracted_x = extract_sensitive_bits(raw_x, bits=bits[i])
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

    # save the model
    # joblib.dump(proxy_net_1, './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/lr.model')


folder = './saved_model/student/0x0040-0x0/scratch/ND6/'
net_paths = [
    folder + '8_bits/0.7185_14_10_5_3_student_6_distinguisher.h5',      # 1
    # folder + '8_bits/0.7221_14_11_5_2_student_6_distinguisher.h5',      # 2
    # folder + '8_bits/0.7179_12_10_4_0_student_6_distinguisher.h5',      # 3
    folder + '8_bits/0.7195_12_9_4_1_student_6_distinguisher.h5',       # 4

    # folder + '8_bits/0.6678_14_9_5_4_student_6_distinguisher.h5',       # 5
    # folder + '8_bits/0.6708_14_10_2_0_student_6_distinguisher.h5',       # 6
]

selected_bits = [
    [14, 13, 12, 11, 10, 5, 4, 3],      # 1
    # [14, 13, 12, 11, 5, 4, 3, 2],       # 2
    # [12, 11, 10, 4, 3, 2, 1, 0],        # 3
    [12, 11, 10, 9, 4, 3, 2, 1],        # 4

    # [14, 13, 12, 11, 10, 9, 5, 4],      # 5
    # [14, 13, 12, 11, 10, 2, 1, 0]       # 6
]

teacher = './saved_model/teacher/0x0040-0x0/6_distinguisher.h5'

# when the first three NDs (1, 2, 3) are used, and the training method is normal,
# the testing acc is  0.739149
# coefs are  [[2.10043672 1.49220071 3.04872297]]
# intercept is  [-3.19441667]

# when the first four NDs (1, 2, 3, 4) are used, and the training method is normal,
# the testing acc is  0.741603
# coefs are  [[2.00549428 1.30865633 1.51041625 1.9889825 ]]
# intercept is  [-3.26305822]
# when the teacher-student scheme is adopted,
# the testing acc is  0.744757
# coefs are  [[4.22554243 2.48937025 2.81402202 4.22461696]]
# intercept is  [-6.98539954]

# when six NDs (1, 2, 3, 4, 5, 6) are used, and the training method is normal,
# the testing acc is  0.747157
# coefs are  [[1.14466772 1.6672402  1.72325925 1.14684741 1.60408066 0.72450344]]
# intercept is  [-3.86156941]

# when two NDs (1, 4) are used, and the training method is normal,
# the testing acc is  0.738363
# coefs are  [[2.85744491 3.62420957]]
# intercept is  [-3.1249105]

train_LR(nr=6, diff=(0x0040, 0), nets=net_paths, bits=selected_bits, teacher=teacher)


# folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_paths = [
#     folder + '0.558_13_10_5_4_student_7_distinguisher.h5',
#     folder + '0.5223_7_2_student_7_distinguisher.h5',
#     # folder + '0.5224_10_8_4_2_student_7_distinguisher.h5',
#     # folder + '0.5393_12_9_5_4_student_7_distinguisher.h5',
#     folder + '0.5397_14_9_student_7_distinguisher.h5',
#     folder + '0.5722_14_11_5_4_student_7_distinguisher.h5'
# ]
#
# selected_bits = [
#     [13, 12, 11, 10, 5, 4],
#     [7, 6, 5, 4, 3, 2],
#     # [10, 9, 8, 4, 3, 2],
#     # [12, 11, 10, 9, 5, 4],
#     [14, 13, 12, 11, 10, 9],
#     [14, 13, 12, 11, 5, 4]
# ]
# model result is:
# the testing acc is  0.589312
# coefs are  [[1.77355501 3.30037086 2.81406491 3.5147626 ]]
# intercept is  [-5.7084619]

# folder = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# net_paths = [
#     folder + '0.558_13_10_5_4_student_7_distinguisher.h5',
#     # folder + '0.5223_7_2_student_7_distinguisher.h5',
#     # folder + '0.5224_10_8_4_2_student_7_distinguisher.h5',
#     # folder + '0.5393_12_9_5_4_student_7_distinguisher.h5',
#     folder + '0.5397_14_9_student_7_distinguisher.h5',
#     folder + '0.5722_14_11_5_4_student_7_distinguisher.h5',
#     folder + '0.523_5_0_student_7_distinguisher.h5',
#     folder + '0.5466_12_9_3_2_student_7_distinguisher.h5',
#     folder + '0.5608_13_10_4_3_student_7_distinguisher.h5',
#     # folder + '0.522_14_13_5_2_student_7_distinguisher.h5'
# ]
#
# selected_bits = [
#     [13, 12, 11, 10, 5, 4],
#     # [7, 6, 5, 4, 3, 2],
#     # [10, 9, 8, 4, 3, 2],
#     # [12, 11, 10, 9, 5, 4],
#     [14, 13, 12, 11, 10, 9],
#     [14, 13, 12, 11, 5, 4],
#     [5, 4, 3, 2, 1, 0],
#     [12, 11, 10, 9, 3, 2],
#     [13, 12, 11, 10, 4, 3],
#     # [14, 13, 5, 4, 3, 2]
# ]
# #
# # the testing acc is  0.590769
# # coefs are  [[0.33108623 1.27736923 3.5088675  1.14009693 1.56519722 1.48802543]]
# # intercept is  [-4.6558662]


# folder_1 = './saved_model/student/0x0040-0x0/scratch/small_net/v1_setting/'
# folder_2 = './saved_model/student/0x0040-0x0/scratch/'
# net_paths = [
#     folder_1 + '0.5913_14_9_5_4_student_7_distinguisher.h5',       # 0
#     # folder_1 + '0.5842_14_10_5_4_student_7_distinguisher.h5',       # 1
#     # folder_2 + '0.5556_12_9_5_2_student_7_distinguisher.h5',        # 2
#     # folder_2 + '0.5346_12_11_5_0_student_7_distinguisher.h5',       # 3
#     # folder_2 + '0.504_14_11_9_6_student_7_distinguisher.h5',        # 4
#     folder_2 + '0.5311_12_11_5_1_student_7_distinguisher.h5',       # 5
# ]
#
# selected_bits = [
#     [14, 13, 12, 11, 10, 9, 5, 4],  # 0
#     # [14, 13, 12, 11, 10, 5, 4],     # 1
#     # [12, 11, 10, 9, 5, 4, 3, 2],    # 2
#     # [12, 11, 5, 4, 3, 2, 1, 0],     # 3
#     # [14, 13, 12, 11, 9, 8, 7, 6],   # 4
#     [12, 11, 5, 4, 3, 2, 1]      # 5
# ]
#
# teacher = './saved_model/teacher/0x0040-0x0/7_distinguisher.h5'

# when the first two NDs (1, 2) are adopted, and the training method is normal,
# the testing acc is  0.5912
# coefs are  [[3.67253141, 2.80168456]]
# intercept is  [-3.23310442]
# the second model is as follows
# the testing acc is  0.592162
# coefs are  [[3.67883782 2.78522049]]
# intercept is  [-3.22849098]

# when the first three NDs (1, 2, 3) are adopted, and the training method is normal,
# the testing acc is  0.592768
# coefs are  [[3.71535209, 2.33765869, 1.42868269]]
# intercept is  [-3.73685329]

# when the first four NDs (1, 2, 3, 4) are adopted, and the training method is normal,
# the testing acc is  0.593442
# coefs are  [[3.70784787, 2,36417626, 1.40795921, -0.65892418]]
# intercept is  [-3.40735885]

# when the first four NDs (1, 2, 4) are adopted, and the teacher-student training method is normal,
# the testing acc is  0.592408
# coefs are  [[22.82977078 18.51521106 -9.60184873]]
# intercept is  [-16.28364789]
# when the training method is normal:
# the testing acc is  0.592131
# coefs are  [[ 3.67272472e+00  2.78330917e+00 -9.01506277e-06]]
# intercept is  [-3.2233376]

# when five NDs (0, 1, 2, 3, 4) are adopted, and the training method is normal,
# the testing acc is  0.596444
# coefs are  [[3.15369626 0.83585576 1.06850478 2.43136906 0.        ]]
# intercept is  [-3.74295534]

# when three NDs (0, 2, 3) are adopted, and the training method is normal,
# the testing acc is  0.596477
# coefs are  [[3.93053784 0.8062921  2.6585548 ]]
# intercept is  [-3.69473792]

# when two NDs (0, 3) are adopted, and the training method is normal,
# the testing acc is  0.597587
# coefs are  [[4.10962769 3.36111323]]
# intercept is  [-3.73423481]

# when two NDs (0, 5) are adopted, and the training method is normal,
# the testing acc is  0.595568
# coefs are  [[4.12507959 3.61160246]]
# intercept is  [-3.86370923]

# train_LR(nr=7, diff=(0x0040, 0), nets=net_paths, bits=selected_bits, teacher=teacher)