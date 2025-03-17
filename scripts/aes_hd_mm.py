import numpy as np
import h5py
from scalib.metrics import SNR
import matplotlib.pyplot as plt
from src.datasets.load_aes_hd_mm import *


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


def get_features(dataset, target_byte: int, n_poi=100, plot_path_name=""):
    snr_val_share_1 = snr_fast(np.array(dataset.x_profiling, dtype=np.int16), np.asarray(dataset.share1_profiling[target_byte, :]))
    snr_val_share_2 = snr_fast(np.array(dataset.x_profiling, dtype=np.int16), np.asarray(dataset.share2_profiling[target_byte, :]))
    snr_val_share_1[np.isnan(snr_val_share_1)] = 0
    snr_val_share_2[np.isnan(snr_val_share_2)] = 0

    ind_snr_masks_poi_sm = np.argsort(snr_val_share_1)[::-1][:int(n_poi / 2)]
    ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)

    ind_snr_masks_poi_r2 = np.argsort(snr_val_share_2)[::-1][:int(n_poi / 2)]
    ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

    poi_profiling = np.concatenate((ind_snr_masks_poi_sm_sorted, ind_snr_masks_poi_r2_sorted), axis=0)

    snr_val_share_1 = snr_fast(np.array(dataset.x_attack, dtype=np.int16), np.asarray(dataset.share1_attack[target_byte, :]))
    snr_val_share_2 = snr_fast(np.array(dataset.x_attack, dtype=np.int16), np.asarray(dataset.share2_attack[target_byte, :]))
    snr_val_share_1[np.isnan(snr_val_share_1)] = 0
    snr_val_share_2[np.isnan(snr_val_share_2)] = 0

    ind_snr_masks_poi_sm = np.argsort(snr_val_share_1)[::-1][:int(n_poi / 2)]
    ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)

    ind_snr_masks_poi_r2 = np.argsort(snr_val_share_2)[::-1][:int(n_poi / 2)]
    ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

    poi_attack = np.concatenate((ind_snr_masks_poi_sm_sorted, ind_snr_masks_poi_r2_sorted), axis=0)

    plt.plot(snr_val_share_1[ind_snr_masks_poi_sm_sorted])
    plt.plot(snr_val_share_2[ind_snr_masks_poi_r2_sorted])
    plt.savefig(plot_path_name)
    plt.close()

    return dataset.x_profiling[:, poi_profiling], dataset.x_attack[:, poi_attack]


dataset_target = ReadAESHDMM(200000, 0, 50000, 0, "ID", '/tudelft.net/staff-umbrella/dlsca/Guilherme/aes_hd_mm.h5', number_of_samples=3250)

figure_filepath = '/tudelft.net/staff-umbrella/dlsca/Guilherme/AES_HD_MM/AES_HD_MM_rpoi/aes_hd_mm_100poi.png'
profiling_traces_rpoi, attack_traces_rpoi = get_features(dataset_target, 4, 100, figure_filepath)
