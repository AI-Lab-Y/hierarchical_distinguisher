import student_net_for_new_inputs as tn
import speck as sp


sp.check_testvector()
# for 7-round distinguishers
# selected_bits = [14, 13, 12, 11, 10, 9, 5, 4]
selected_bits = [12, 11, 5, 4, 3, 2, 1]

# for 6-round distinguishers
# selected_bits = [14, 13, 12, 11, 10, 5, 4, 3]
# selected_bits = [12, 11, 10, 9, 4, 3, 2, 1]

nr = 7
model_folder = './saved_model/new_inputs/'
tn.train_speck_distinguisher(10, num_rounds=nr, depth=1, diff=(0x0040, 0), bits=selected_bits, folder=model_folder)