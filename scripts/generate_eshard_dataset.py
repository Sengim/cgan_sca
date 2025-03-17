import numpy as np
import matplotlib.pyplot as plt
import h5py
from src.datasets.load_eshard import *


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


""" 
Read ESHARD AES Dataset from npy files and generate h5 dataset 
"""

target_byte = 0

n_profiling = 90000
n_attack = 10000

samples = np.load("/tudelft.net/staff-umbrella/dlsca/Eshard/traces.npy")
plaintexts = np.load("/tudelft.net/staff-umbrella/dlsca/Eshard/plaintext.npy")
masks = np.load("/tudelft.net/staff-umbrella/dlsca/Eshard/mask.npy")
keys = np.load("/tudelft.net/staff-umbrella/dlsca/Eshard/key.npy")

profiling_traces = np.array(samples[:n_profiling], dtype="float32")
attack_traces = np.array(samples[n_profiling:], dtype="float32")

profiling_plaintexts = plaintexts[:n_profiling]
profiling_keys = keys[:n_profiling]
profiling_masks = masks[:n_profiling]

attack_plaintexts = plaintexts[n_profiling:]
attack_keys = keys[n_profiling:]
attack_masks = masks[n_profiling:]

out_file = h5py.File('/tudelft.net/staff-umbrella/dlsca/Guilherme/eshard.h5', 'w')

profiling_index = [n for n in range(n_profiling)]
attack_index = [n for n in range(n_attack)]

profiling_traces_group = out_file.create_group("Profiling_traces")
attack_traces_group = out_file.create_group("Attack_traces")

profiling_traces_group.create_dataset(name="traces", data=profiling_traces, dtype=profiling_traces.dtype)
attack_traces_group.create_dataset(name="traces", data=attack_traces, dtype=attack_traces.dtype)

metadata_type_profiling = np.dtype([("plaintext", profiling_plaintexts.dtype, (len(profiling_plaintexts[0]),)),
                                    ("key", profiling_keys.dtype, (len(profiling_keys[0]),)),
                                    ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))
                                    ])
metadata_type_attack = np.dtype([("plaintext", attack_plaintexts.dtype, (len(attack_plaintexts[0]),)),
                                 ("key", attack_keys.dtype, (len(attack_keys[0]),)),
                                 ("masks", attack_masks.dtype, (len(attack_masks[0]),))
                                 ])

profiling_metadata = np.array([(profiling_plaintexts[n], profiling_keys[n], profiling_masks[n]) for n in profiling_index],
                              dtype=metadata_type_profiling)
profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

attack_metadata = np.array([(attack_plaintexts[n], attack_keys[n], attack_masks[n]) for n in attack_index], dtype=metadata_type_attack)
attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

out_file.flush()
out_file.close()

""" 
Read ESHARD AES Dataset labeled with Identity Leakage Model 
"""

dataset_target = ReadEshard(90000, 0, 10000, target_byte, "ID", '/tudelft.net/staff-umbrella/dlsca/Guilherme/eshard.h5',
                            number_of_samples=1400)

profiling_traces_rpoi, attack_traces_rpoi = get_features(dataset_target, target_byte, 100,
                                                         '/tudelft.net/staff-umbrella/dlsca/Guilherme/ESHARD/ESHARD_rpoi/eshard_100poi.png')
out_file = h5py.File('/tudelft.net/staff-umbrella/dlsca/Guilherme/ESHARD/ESHARD_rpoi/eshard_100poi.h5', 'w')

profiling_index = [n for n in range(n_profiling)]
attack_index = [n for n in range(n_attack)]

profiling_traces_group = out_file.create_group("Profiling_traces")
attack_traces_group = out_file.create_group("Attack_traces")

profiling_traces_group.create_dataset(name="traces", data=profiling_traces_rpoi, dtype=profiling_traces_rpoi.dtype)
attack_traces_group.create_dataset(name="traces", data=attack_traces_rpoi, dtype=attack_traces_rpoi.dtype)

metadata_type_profiling = np.dtype([("plaintext", profiling_plaintexts.dtype, (len(profiling_plaintexts[0]),)),
                                    ("key", profiling_keys.dtype, (len(profiling_keys[0]),)),
                                    ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))
                                    ])
metadata_type_attack = np.dtype([("plaintext", attack_plaintexts.dtype, (len(attack_plaintexts[0]),)),
                                 ("key", attack_keys.dtype, (len(attack_keys[0]),)),
                                 ("masks", attack_masks.dtype, (len(attack_masks[0]),))
                                 ])

profiling_metadata = np.array([(profiling_plaintexts[n], profiling_keys[n], profiling_masks[n]) for n in profiling_index],
                              dtype=metadata_type_profiling)
profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

attack_metadata = np.array([(attack_plaintexts[n], attack_keys[n], attack_masks[n]) for n in attack_index], dtype=metadata_type_attack)
attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

out_file.flush()
out_file.close()

""" 
Read ESHARD AES Dataset labeled with Hamming Weight Leakage Model
"""

dataset_target = ReadEshard(90000, 0, 10000, target_byte, "HW", '/tudelft.net/staff-umbrella/dlsca/Guilherme/eshard.h5',
                            number_of_samples=1400)

profiling_traces_rpoi_hw, attack_traces_rpoi_hw = get_features(dataset_target, target_byte, 100,
                                                               '/tudelft.net/staff-umbrella/dlsca/Guilherme/ESHARD/ESHARD_rpoi/eshard_100poi_hw.png')
out_file = h5py.File('/tudelft.net/staff-umbrella/dlsca/Guilherme/ESHARD/ESHARD_rpoi/eshard_100poi_hw.h5', 'w')

profiling_index = [n for n in range(n_profiling)]
attack_index = [n for n in range(n_attack)]

profiling_traces_group = out_file.create_group("Profiling_traces")
attack_traces_group = out_file.create_group("Attack_traces")

profiling_traces_group.create_dataset(name="traces", data=profiling_traces_rpoi_hw, dtype=profiling_traces_rpoi_hw.dtype)
attack_traces_group.create_dataset(name="traces", data=attack_traces_rpoi_hw, dtype=attack_traces_rpoi_hw.dtype)

metadata_type_profiling = np.dtype([("plaintext", profiling_plaintexts.dtype, (len(profiling_plaintexts[0]),)),
                                    ("key", profiling_keys.dtype, (len(profiling_keys[0]),)),
                                    ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))
                                    ])
metadata_type_attack = np.dtype([("plaintext", attack_plaintexts.dtype, (len(attack_plaintexts[0]),)),
                                 ("key", attack_keys.dtype, (len(attack_keys[0]),)),
                                 ("masks", attack_masks.dtype, (len(attack_masks[0]),))
                                 ])

profiling_metadata = np.array([(profiling_plaintexts[n], profiling_keys[n], profiling_masks[n]) for n in profiling_index],
                              dtype=metadata_type_profiling)
profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

attack_metadata = np.array([(attack_plaintexts[n], attack_keys[n], attack_masks[n]) for n in attack_index], dtype=metadata_type_attack)
attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

out_file.flush()
out_file.close()
