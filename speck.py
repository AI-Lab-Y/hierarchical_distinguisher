import numpy as np
from os import urandom


def WORD_SIZE():
    return(16)


def ALPHA():
    return(7)


def BETA():
    return(2)


MASK_VAL = 2 ** WORD_SIZE() - 1


def shuffle_together(l):
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)


def rol(x, k):
    # x = x & 0x00ffffff      # set 24~31 bits to 0
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))


def ror(x, k):
    # x = x & 0x00ffffff      # set 24~31 bits to 0
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return(c0,c1)


def dec_one_round(c,k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return(c0, c1)


def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return(ks)


def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return(x,y)


def check_testvector():
  key = (0x1918, 0x1110, 0x0908, 0x0100)
  pt = (0x6574, 0x694c)
  ks = expand_key(key, 22)
  ct = encrypt(pt, ks)
  if (ct == (0xa868, 0x42f2)):
    print("Testvector verified.")
    return(True)
  else:
    print("Testvector not verified.")
    return(False)


#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((4 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
  for i in range(4 * WORD_SIZE()):
    index = i // WORD_SIZE()
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose()
  return(X)


# baseline training data generator,  speck48/72
def make_train_data(n, nr, diff=(0x0040, 0)):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1
  keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
  plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
  plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]
  num_rand_samples = np.sum(Y==0)
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples), dtype=np.uint16)
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples), dtype=np.uint16)
  ks = expand_key(keys, nr)
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
  return(X,Y)


def convert_to_binary_for_new_input(arr):
  X = np.zeros((3 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
  for i in range(3 * WORD_SIZE()):
    index = i // WORD_SIZE()
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose()
  return(X)


def extract_sensitive_bits_for_new_input(raw_x, bits=[14,13,12,11,10,9,8,7]):
    # get new-x according to sensitive bits
    id0 = [15 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + 16 * i for i in range(3) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]

    return new_x


# new input: (c0l ^ c1l, c0r ^ c1r, c0l ^ c0r)
def make_data_with_new_input(n, nr, diff=(0x0040, 0), selected_bits=None):
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
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)

    t0, t1, t2 = ctdata0l ^ ctdata1l, ctdata0r ^ ctdata1r, ctdata0l ^ ctdata0r
    raw_X = convert_to_binary_for_new_input([t0, t1, t2])
    X = extract_sensitive_bits_for_new_input(raw_x=raw_X, bits=selected_bits)
    return (X, Y)


# check_testvector()
