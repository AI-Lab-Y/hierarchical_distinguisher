The details of the Hierarchical distinguisher:

diff = (0x0040, 0x0)
folder = './saved_model/new_inputs/'

# # for 7 round distinguisher
# nr = 7
# net_paths = [
#     folder + '0.5886_14_9_5_4_nd7.npy',       # 0
#     folder + '0.538_12_11_5_1_nd7.npy'       # 5
# ]
# nets = []
# for net_path in net_paths:
#     nets.append(np.load(net_path))
#
# # [w_0, w_1 ...]
# weights = [-3.7528234, 4.08646331, 3.4168975]
#
# selected_bits = [
#     [14, 13, 12, 11, 10, 9, 5, 4],  # 0
#     [12, 11, 5, 4, 3, 2, 1]      # 5
# ]

# for 6 round distinguisher
nr = 6
net_paths = [
    folder + '0.7173_14_10_5_3_nd6.npy',      # 1
    folder + '0.7206_12_9_4_1_nd6.npy'       # 4
]
nets = []
for net_path in net_paths:
    nets.append(np.load(net_path))
# [w_0, w_1 ...]
weights = [-3.13645595, 2.82072909, 3.64216749]
selected_bits = [
    [14, 13, 12, 11, 10, 5, 4, 3],      # 1
    [12, 11, 10, 9, 4, 3, 2, 1],        # 4
]