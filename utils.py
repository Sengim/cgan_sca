import numpy as np
from datetime import datetime
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from src.datasets.paths import *
from src.datasets.load_ascadr import *
from src.datasets.load_ascadf import *
from src.datasets.load_dpav42 import *
from src.datasets.load_eshard import *
from src.datasets.load_chesctf import *
from os.path import exists


def snr_fast(x, y):
    ns = x.shape[1]
    unique = np.unique(y)
    means = np.zeros((len(unique), ns))
    variances = np.zeros((len(unique), ns))

    for i, u in enumerate(unique):
        new_x = x[np.argwhere(y == int(u))]
        means[i] = np.mean(new_x, axis=0)
        variances[i] = np.var(new_x, axis=0)
    return np.var(means, axis=0) / np.mean(variances, axis=0)

def get_features_bit(dataset, target_byte: int, n_poi=100):
    poi = np.zeros(n_poi, dtype=np.int32)
    per_bit =  n_poi//16
    for i in range(8):
        poi[i*per_bit: (i+1)*per_bit] = get_features_bit_per(np.array(dataset.x_profiling[:20000], dtype=np.int16), np.asarray(dataset.share1_profiling[target_byte, :20000], dtype=np.uint8), i,per_bit)
        poi[n_poi//2 + i*per_bit:n_poi//2 + (i+1)*per_bit] = get_features_bit_per(np.array(dataset.x_profiling[:20000], dtype=np.int16), np.asarray(dataset.share2_profiling[target_byte, :20000], dtype=np.uint8), i,per_bit)
    return dataset.x_profiling[:, poi], dataset.x_attack[:, poi]


def get_features(dataset, target_byte: int, n_poi=100):
    snr_arr = get_snr_shares(dataset, target_byte)
    snr_val_share_1 = snr_arr[0]
    snr_val_share_2 = snr_arr[1]
    snr_val_share_1[np.isnan(snr_val_share_1)] = 0
    snr_val_share_2[np.isnan(snr_val_share_2)] = 0
    
    ind_snr_masks_poi_sm = np.argsort(snr_val_share_1)[::-1][:int(n_poi / 2)]
    sorted_poi_masks_sm = np.argsort(snr_val_share_1)[::-1]
    ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)

    ind_snr_masks_poi_r2 = np.argsort(snr_val_share_2)[::-1][:int(n_poi / 2)]
    sorted_poi_masks_r2 = np.argsort(snr_val_share_2)[::-1]
    ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

    poi_profiling = np.concatenate((ind_snr_masks_poi_sm_sorted, ind_snr_masks_poi_r2_sorted), axis=0)

    return dataset.x_profiling[:, poi_profiling], dataset.x_attack[:, poi_profiling]

def get_features_order(dataset, target_byte, n_poi):
    snr_arr = get_snr_shares(dataset, target_byte)
    order = snr_arr.shape[0]
    poi_profiling = None
    for i in range(order):
        snr_val = snr_arr[i]
        snr_val[np.isnan(snr_val)] = 0
        ind_snr_masks_poi_sm = np.argsort(snr_val)[::-1][:int(n_poi //order)]
        ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)
        poi_profiling = ind_snr_masks_poi_sm_sorted if i ==0 else np.concatenate((poi_profiling, ind_snr_masks_poi_sm_sorted), axis=0)
    return dataset.x_profiling[:, poi_profiling], dataset.x_attack[:, poi_profiling]




def get_lda_features(dataset, target_byte: int, n_components=10):
    order = 1 if not  (dataset.name=="simulate" or dataset.name =="spook_sw3") else dataset.order
    order = order + 1
    n_poi_snr = min(100, dataset.ns//order)
    x_prof, x_att = get_features(dataset, target_byte, n_poi=n_poi_snr*order)

    result_prof, result_att = None, None
    print(n_components//(order))
    for i in range(order):
        lda = LinearDiscriminantAnalysis(n_components=n_components//(order))
        if dataset.name == "spook_sw3":
            lda.fit(x_prof[:20000, i*n_poi_snr:(i+1)*n_poi_snr], dataset.profiling_shares[ :20000,target_byte,  i])
        elif dataset.name == "simulate":
            lda.fit(x_prof[:20000, i*n_poi_snr:(i+1)*n_poi_snr], dataset.profiling_shares[ :20000,  i])
        else:
             temp = dataset.share1_profiling[target_byte, :20000] if i == 0 else dataset.share2_profiling[target_byte, :20000]
             lda.fit(x_prof[:20000, i*n_poi_snr:(i+1)*n_poi_snr], temp)
        
        s1_prof = lda.transform(x_prof[:, i*n_poi_snr:(i+1)*n_poi_snr] )
        s1_att = lda.transform(x_att[:, i*n_poi_snr:(i+1)*n_poi_snr] )
        result_prof = s1_prof if i == 0 else np.append(result_prof, s1_prof, axis=1)
        result_att = s1_att if i == 0 else np.append(result_att, s1_att, axis=1)
    return result_prof, result_att

def get_pca_features(dataset, target_byte: int, n_components=10):

    order = 1 if not  (dataset.name=="simulate" or dataset.name =="spook_sw3") else dataset.order
    order = order + 1
    n_poi_snr = min(100, dataset.ns//order)
    x_prof, x_att = get_features(dataset, target_byte, n_poi=n_poi_snr*order)

    result_prof, result_att = None, None
    for i in range(order):
        pca = PCA(n_components=n_components//(order))
        pca.fit(x_prof[:20000, i*n_poi_snr:(i+1)*n_poi_snr])
        s1_prof = pca.transform(x_prof[:, i*n_poi_snr:(i+1)*n_poi_snr] )
        s1_att = pca.transform(x_att[:, i*n_poi_snr:(i+1)*n_poi_snr] )
        result_prof = s1_prof if i == 0 else np.append(result_prof, s1_prof, axis=1)
        result_att = s1_att if i == 0 else np.append(result_att, s1_att, axis=1)
    return result_prof, result_att




def create_directory_results(args, path):
    now = datetime.now()
    now_str = f"{now.strftime('%d_%m_%Y_%H_%M_%S')}_{np.random.randint(1000000, 10000000)}"
    dir_results = f"{path}/{args['dataset_reference']}_vs_{args['dataset_target']}_{now_str}"
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)
    return dir_results


def get_features_bit_per(x, y, bit, points):
    temp = snr_fast(x, (y>>(7-bit)) & 1)
    ind_snr_masks_poi_sm = np.argsort(temp)[::-1][:points]
    ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)
    return ind_snr_masks_poi_sm_sorted

def get_snr_shares(dataset, target_byte):
    if dataset.name == "simulate":
        return get_snr_shares_sim(dataset)
    elif dataset.name =="spook_sw3":
        return get_snr_shares_spook(dataset, target_byte)
    result_arr = np.zeros((2, dataset.ns))
    result_arr[0, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.share1_profiling[target_byte, :min(dataset.x_profiling.shape[0], 20000)]))
    result_arr[1, :] = snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.share2_profiling[target_byte, :min(dataset.x_profiling.shape[0], 20000)]))
    return result_arr

def get_snr_shares_sim(dataset):
    result_arr = np.zeros((dataset.order+ 1, dataset.ns))
    order = dataset.order + 1
    for i in range(order):
        result_arr[i, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.profiling_shares[ :min(dataset.x_profiling.shape[0], 20000), i]))
    return result_arr





def get_snr_shares_spook(dataset, target_byte):
    result_arr = np.zeros((dataset.order+1, dataset.ns))
    order = dataset.order+1
    for i in range(order):
        result_arr[i, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.prof_shares[ :min(dataset.x_profiling.shape[0], 20000), target_byte, i]))
    return result_arr


def load_dataset(identifier: str, path: str, target_byte: int, traces_dim: int, leakage_model="ID", num_features=-1):
    
    dataset_file = get_dataset_filepath(path, identifier, traces_dim, leakage_model=leakage_model)
    snr_shortcut = f'{path}/paper_9_gan_features/selected_{num_features}_features_snr_{identifier}_{traces_dim}.h5'
    if num_features > 0 and exists(snr_shortcut):
        dataset_file = snr_shortcut
        traces_dim = num_features
        
    if identifier == "ascad-variable":
        dataset = ReadASCADr(200000, 0, 10000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "ASCAD":
        dataset = ReadASCADf(50000, 0, 10000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "dpa_v42":
        dataset = ReadDPAV42(70000, 0, 5000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "ches_ctf":
        dataset = ReadCHESCTF(45000, 0, 5000, target_byte, leakage_model,
                                                         dataset_file,
                                                         number_of_samples=traces_dim)
    return dataset



def scale_dataset(prof_set, attack_set, scaler):
        prof_new = scaler.fit_transform(prof_set)
        if attack_set is not None:
            attack_new = scaler.transform(attack_set)
        else:
            attack_new = None
        return prof_new, attack_new