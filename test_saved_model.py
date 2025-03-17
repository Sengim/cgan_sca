import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from random_class_mlp_cnn import *
from profiling_and_attack import attack

cgan_prof = sys.argv[2]
#Best model n_prof_cgan training
folders = {10000: "ascad-variable_vs_dpa_v42_10000_29_08_2023_21_28_01_4692279", 
           20000: "ascad-variable_vs_dpa_v42_20000_29_08_2023_15_14_37_6191280",
           30000: "ascad-variable_vs_dpa_v42_30000_29_08_2023_10_42_22_5847420",
           40000: "ascad-variable_vs_dpa_v42_40000_29_08_2023_08_22_58_8775106",
           50000: "ascad-variable_vs_dpa_v42_50000_29_08_2023_04_19_20_6731057",
           60000: "ascad-variable_vs_dpa_v42_60000_28_08_2023_11_06_41_7636267" }



dataset_id = 'dpa_v42'
resample_window=80
traces_dim = 7500

class HackyDatasetObject(object):
    pass

n_prof= int(sys.argv[1])
target_byte = 2
path = "C:/Users/Sengim/Datasets"
dataset = load_dataset(dataset_id, path, target_byte, traces_dim, leakage_model="ID")
dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling[:n_prof], dataset.x_attack, StandardScaler())
dataset.profiling_labels = dataset.profiling_labels[:n_prof]
best_nt = 2000
best_ge = 256
best_hp = None
best_pi = -500
### {'batch_size': 400, 'layers': 4, 'neurons': 10, 'activation': 'relu', 'learning_rate': 0.0001, 'optimizer': 'Adam', 'kernel_initializer': 'glorot_uniform', 'regularization': 'none'}
temp = None
features_prof, features_att = None, None
model = None
if cgan_prof == "wb":
    dataset.x_profiling, dataset.x_attack = get_features(dataset=dataset, target_byte=target_byte, n_poi=100)
elif cgan_prof == "bb":
    pass
else:
    folder = f"C:/Users/Sengim/Datasets/vary_prof_search/{folders[int(cgan_prof)]}"
    model = tf.keras.models.load_model(f"{folder}/generator_{traces_dim}_25000_epoch_199.h5")

    dataset.x_profiling = np.array(model.predict([dataset.x_profiling]))
    dataset.x_attack = np.array(model.predict([dataset.x_attack]))
#temp = {'batch_size': 400, 'layers': 4, 'neurons': 10, 'activation': 'relu', 'learning_rate': 0.0001, 'optimizer': 'Adam', 'kernel_initializer': 'glorot_uniform', 'regularization': 'none'}
#temp = {'batch_size': 1000, 'layers': 1, 'neurons': 50, 'activation': 'relu', 'learning_rate': 0.005, 'optimizer': 'Adam', 'kernel_initializer': 'glorot_uniform', 'regularization': 'none'}
broad_dataset = HackyDatasetObject()
broad_dataset.dataset_target = dataset
for i in range(100):
    attack_mod, seed,hp =  mlp_random(256, traces_dim if cgan_prof=="bb" else 100, hp=temp)
    print(i)
    ge, nt, pi, ge_v = attack(broad_dataset, model, 100, attack_model=attack_mod, batch_size=hp['batch_size'], 
                        original_traces=True, synthetic_traces=False)
    
    if pi > best_pi: 
        print(f"----------------------------------------------------------------------------")
        best_hp = hp
        best_nt = nt
        best_ge = ge
        best_pi = pi
        print(f"PI: {pi}")
        print(hp)
        print(f"----------------------------------------------------------------------------")
    # if (ge <= best_ge or ge < 2) and nt <= best_nt:
    #     print(f"----------------------------------------------------------------------------")
    #     best_hp = hp
    #     best_nt = nt
    #     best_ge = ge
    #     print(f"GE: {best_ge} and NT: {best_nt}")
    #     print(hp)
    #     print(f"----------------------------------------------------------------------------")
np.savez(f"results/hp_search_{n_prof}_{cgan_prof}.npz", best_hp = best_hp, pi= best_pi, best_ge =best_ge, best_nt = best_nt)
print(f"GE: {best_ge} and NT: {best_nt}, Best PI: {best_pi}")
print(best_hp)
