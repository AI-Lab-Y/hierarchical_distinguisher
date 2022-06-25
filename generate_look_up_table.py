import numpy as np
from keras.models import load_model


# x shape: [n]
def uint_to_array(x, bit_len=16):
    y = np.zeros((len(x), bit_len), dtype=np.uint8)
    for j in range(bit_len):
        y[:, j] = (x >> (bit_len - 1 - j)) & 1
    return y


def generate_look_up_table(net_path=None, block_size=16):
    # load neural distinguisher
    nd = load_model(net_path)
    # get inference table
    x = np.array(range(2**block_size), dtype=np.uint32)
    new_x = uint_to_array(x, bit_len=block_size)
    y = nd.predict(new_x, batch_size=10**4, verbose=0)
    y = np.squeeze(y)
    return y


def generate_lookup_tables_for_key_recovery():
    saved_folder = './saved_model/new_inputs/'
    net_path = saved_folder + '0.538_12_11_5_1_student_7_distinguisher.h5'
    y = generate_look_up_table(net_path=net_path, block_size=21)
    np.save(saved_folder+'0.538_12_11_5_1_nd7.npy', y)

    saved_folder = './saved_model/new_inputs/'
    net_path = saved_folder + '0.5886_14_9_5_4_student_7_distinguisher.h5'
    y = generate_look_up_table(net_path=net_path, block_size=24)
    np.save(saved_folder+'0.5886_14_9_5_4_nd7.npy', y)

    saved_folder = './saved_model/new_inputs/'
    net_path = saved_folder + '0.7173_14_10_5_3_student_6_distinguisher.h5'
    y = generate_look_up_table(net_path=net_path, block_size=24)
    np.save(saved_folder+'0.7173_14_10_5_3_nd6.npy', y)

    saved_folder = './saved_model/new_inputs/'
    net_path = saved_folder + '0.7206_12_9_4_1_student_6_distinguisher.h5'
    y = generate_look_up_table(net_path=net_path, block_size=24)
    np.save(saved_folder+'0.7206_12_9_4_1_nd6.npy', y)