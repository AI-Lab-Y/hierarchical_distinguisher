import numpy as np


class HD(object):
    def __init__(self, nets, bits, weights):
        assert len(nets) + 1 == len(weights)
        self.nets = nets
        self.bits = bits
        self.weights = weights

    @staticmethod
    def extract_sensitive_bits_for_new_input(x, bits):
        # get new-x according to sensitive bits
        id0 = [15 - v for v in bits]
        id1 = [v + 16 * i for i in range(3) for v in id0]
        new_x = x[:, id1]
        return new_x

    @staticmethod
    def array_to_uint(x, bit_len):
        assert bit_len < 32
        y = np.zeros(len(x), dtype=np.uint32)
        for j in range(bit_len):
            y += x[:, j] * (1 << (bit_len - 1 - j))
        return y

    def predict(self, x):
        n = x.shape[0]
        z_total = np.zeros(n, dtype=np.float)
        num = len(self.nets)
        for j in range(num):
            nd = self.nets[j]
            selected_bits = self.bits[j]
            w = self.weights[j + 1]
            selected_X = self.extract_sensitive_bits_for_new_input(x, bits=selected_bits)
            z_tmp = nd[self.array_to_uint(selected_X, bit_len=len(selected_bits) * 3)]
            z_tmp = z_tmp * w
            z_total += z_tmp
        z_total += self.weights[0]
        return z_total
