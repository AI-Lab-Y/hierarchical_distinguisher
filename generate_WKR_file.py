import numpy as np
from os import urandom
import speck as sp


def extract_sensitive_bits_for_new_input(raw_x, bits=[14,13,12,11,10,9,8,7]):
    # get new-x according to sensitive bits
    id0 = [15 - v for v in bits]
    id1 = [v + 16 * i for i in range(3) for v in id0]
    new_x = raw_x[:, id1]
    return new_x


# x shape: [n]
def uint_to_array(x, bit_len=16):
    y = np.zeros((len(x), bit_len), dtype=np.uint8)
    for j in range(bit_len):
        y[:, j] = (x >> (bit_len - 1 - j)) & 1
    return y


# x shape: [n]
def array_to_uint(x, bit_len=16):
    y = np.zeros(len(x), dtype=np.uint32)
    for j in range(bit_len):
        y += x[:, j] * (1 << (bit_len - 1 - j))
    return y


# generate WKR file for HDs
# ND‘s input is x^1: (c0l ^ c1l, c0r ^ c1r, c0l ^ c0r)
def wrong_key_decryption(n, diff=(0x0040, 0x0), nr=7, nets=None, bits=None, weights=None):
    assert len(nets) + 1 == len(weights)
    means = np.zeros(2**16); sig = np.zeros(2**16)
    for i in range(2**16):
        print('i is {}'.format(i))
        keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(keys, nr+1)
        pt0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
        pt0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
        pt1l, pt1r = pt0l ^ diff[0], pt0r ^ diff[1]
        ct0l, ct0r = sp.encrypt((pt0l, pt0r), ks)
        ct1l, ct1r = sp.encrypt((pt1l, pt1r), ks)
        rsubkeys = i ^ ks[nr]
        c0l, c0r = sp.dec_one_round((ct0l, ct0r), rsubkeys)
        c1l, c1r = sp.dec_one_round((ct1l, ct1r), rsubkeys)
        t0, t1, t2 = c0l ^ c1l, c0r ^ c1r, c0l ^ c0r
        X = sp.convert_to_binary_for_new_input([t0, t1, t2])

        # obtain the output of the hierarchical distinguisher
        z_total = np.zeros(n, dtype=np.float)
        num = len(nets)
        for j in range(num):
            nd = nets[j]
            selected_bits = bits[j]
            w = weights[j+1]
            selected_X = sp.extract_sensitive_bits_for_new_input(X, bits=selected_bits)
            z_tmp = nd[array_to_uint(selected_X, bit_len=len(selected_bits)*3)]
            z_tmp = z_tmp * w
            z_total += z_tmp
        z_total += weights[0]
        means[i] = np.mean(z_total)
        sig[i] = np.std(z_total)
    return(means, sig)


n = 2000
diff = (0x0040, 0x0)
folder = './saved_model/new_inputs/'

# for 7 round distinguisher
nr = 7
net_paths = [
    folder + '0.5886_14_9_5_4_nd7.npy',       # 0
    folder + '0.538_12_11_5_1_nd7.npy'       # 5
]
nets = []
for net_path in net_paths:
    nets.append(np.load(net_path))

# [w_0, w_1 ...]
weights = [-3.7528234, 4.08646331, 3.4168975]

selected_bits = [
    [14, 13, 12, 11, 10, 9, 5, 4],  # 0
    [12, 11, 5, 4, 3, 2, 1]      # 5
]

# # for 6 round distinguisher
# nr = 6
# net_paths = [
#     folder + '0.7173_14_10_5_3_nd6.npy',      # 1
#     folder + '0.7206_12_9_4_1_nd6.npy'       # 4
# ]
# nets = []
# for net_path in net_paths:
#     nets.append(np.load(net_path))
# # [w_0, w_1 ...]
# weights = [-3.13645595, 2.82072909, 3.64216749]
# selected_bits = [
#     [14, 13, 12, 11, 10, 5, 4, 3],      # 1
#     [12, 11, 10, 9, 4, 3, 2, 1],        # 4
# ]

mean_res, std_res = wrong_key_decryption(n=n, diff=diff, nr=nr, nets=nets,
                                         bits=selected_bits, weights=weights)
np.save(folder + 'data_wrong_key_{}r_mean.npy'.format(nr), mean_res)
np.save(folder + 'data_wrong_key_{}r_std.npy'.format(nr), std_res)