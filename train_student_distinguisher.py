import student_net as tn
import speck as sp


sp.check_testvector()
# selected_bits = [14, 13, 12, 11, 10, 9, 5, 4, 3, 2, 1, 0]   # acc = 0.7796
# selected_bits = [14, 13, 12, 11, 10, 4, 3, 2, 1, 0]   # acc = 0.73468
# selected_bits = [14, 13, 12, 11, 10, 5, 4, 3, 2, 1, 0]   # acc = 0.7671
# selected_bits = [14, 13, 12, 11, 10, 5, 4, 3, 2, 1]     # acc = 0.7604
# for 8-bits NDs
# selected_bits = [14, 13, 12, 11, 10, 5, 4, 3]    # acc = 0.7185
# selected_bits = [14, 13, 12, 11, 5, 4, 3, 2]    # acc = 0.722136
# selected_bits = [12, 11, 10, 4, 3, 2, 1, 0]    # acc = 0.7179
# selected_bits = [10, 9, 5, 4, 3, 2, 1, 0]    # acc = 0.6968
# selected_bits = [12, 11, 10, 9, 4, 3, 2, 1]    # acc = 0.7195
# selected_bits = [14, 13, 12, 11, 10, 2, 1, 0]    # acc = 0.6708
# selected_bits = [14, 13, 12, 11, 10, 9, 5, 4]    # acc = 0.6678
#
# nr = 6
# model_folder = './saved_model/student/0x0040-0x0/scratch/ND6/8_bits/'
# tn.train_speck_distinguisher(10, num_rounds=nr, depth=1, diff=(0x0040, 0), bits=selected_bits, folder=model_folder)


# selected_bits = [15, 14, 13, 12, 11, 10, 9, 8]   # acc = 0.5421
# selected_bits = [7, 6, 5, 4, 3, 2, 1, 0]    # acc = 0.5224
# selected_bits = [15, 14, 13, 12, 7, 6, 5, 4]    # acc = 0.5047
# selected_bits = [11, 10, 9, 8, 3, 2, 1, 0]    # acc = 0.5223
# selected_bits = [9, 8, 7, 6, 2, 1, 0]    # acc = 0.5071
# selected_bits = [14, 13, 12, 11, 10, 5, 4]    # acc = 0.5842
# selected_bits = [12, 11, 10, 9, 5, 4, 3, 2]    # acc = 0.5556
# selected_bits = [12, 11, 5, 4, 3, 2, 1, 0]    # acc = 0.53459299998
# selected_bits = [14, 13, 12, 11, 9, 8, 7, 6]    # acc = 0.5040
selected_bits = [12, 11, 5, 4, 3, 2, 1]           # acc = 0.5311


nr = 7
model_folder = './saved_model/student/0x0040-0x0/scratch/'
tn.train_speck_distinguisher(10, num_rounds=nr, depth=1, diff=(0x0040, 0), bits=selected_bits, folder=model_folder)

