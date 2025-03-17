import h5py
import numpy as np
from tqdm import tqdm
import random
from numba import njit
from utils import *
import matplotlib.pyplot as plt

target_byte = 0


def load_profiling_ascad_v2(ascad_database_file, resample_factor=2, train_begin=0, profiling_traces=50000, attack_label='sbox_masked'):
    # 'alpha_mask': affine mask
    # 'beta_mask': boolean mask
    # 'sbox_masked': multGF256(alpha,Sbox(p[permInd[i]]^k[permInd[i]]))^beta for each i in [0..15]
    # 'sbox_masked_with_perm': multGF256(alpha,Sbox(p[i]^k[i]))^beta for each i in [0..15]
    # 'perm_index': permuted indice value permInd[i] for each i in [0..15]
    train_end = train_begin + profiling_traces

    in_file = h5py.File(f'{ascad_database_file}/ascadv2-extracted.h5', "r")

    X_profiling = in_file['Profiling_traces/traces'][train_begin:train_end][:, ::resample_factor]
    Y_profiling = np.array(in_file['Profiling_traces/labels'][:]['sbox_masked'][train_begin:train_end], dtype=np.uint8)
    perm_index = np.array(in_file['Profiling_traces/labels'][:]['perm_index'][train_begin:train_end], dtype=np.uint8)

    # We load the plaintext with known permutation
    P_profiling_perm = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'][train_begin:train_end], dtype=np.uint8)
    P_profiling = P_profiling_perm[np.arange(P_profiling_perm.shape[0])[:, None], perm_index]

    # We load the key with known permutation
    K_profiling_perm = np.array(in_file['Profiling_traces/metadata'][:]['key'][train_begin:train_end], dtype=np.uint8)
    K_profiling = K_profiling_perm[np.arange(K_profiling_perm.shape[0])[:, None], perm_index]

    # We simulate a fix key (the orginal key is random)
    key_init = np.array([77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105], dtype=np.uint8)

    simulated_key = np.repeat(key_init[np.newaxis, :], P_profiling.shape[0], axis=0)
    simulated_P_profiling = P_profiling ^ K_profiling ^ simulated_key

    mul_mask = np.array(in_file['Profiling_traces/labels'][:]['alpha_mask'][train_begin:train_end], dtype=np.uint8)
    add_mask = np.array(in_file['Profiling_traces/labels'][:]['beta_mask'][train_begin:train_end], dtype=np.uint8)

    return X_profiling, simulated_P_profiling, simulated_key, Y_profiling, mul_mask, add_mask, perm_index, P_profiling_perm, K_profiling_perm


def load_attack_ascad_v2(ascad_database_file, resample_factor=2, train_begin=0, attack_traces=50000, attack_label='sbox_masked'):
    # 'alpha_mask': affine mask
    # 'beta_mask': boolean mask
    # 'sbox_masked': multGF256(alpha,Sbox(p[permInd[i]]^k[permInd[i]]))^beta for each i in [0..15]
    # 'sbox_masked_with_perm': multGF256(alpha,Sbox(p[i]^k[i]))^beta for each i in [0..15]
    # 'perm_index': permuted indice value permInd[i] for each i in [0..15]
    train_end = train_begin + attack_traces

    in_file = h5py.File(f'{ascad_database_file}/ascadv2-extracted.h5', "r")

    print(in_file['Attack_traces/traces'])
    X_attack = in_file['Attack_traces/traces'][train_begin:train_end][:, ::resample_factor]
    Y_attack = np.array(in_file['Attack_traces/labels'][:]['sbox_masked'][train_begin:train_end], dtype=np.uint8)
    perm_index = np.array(in_file['Attack_traces/labels'][:]['perm_index'][train_begin:train_end], dtype=np.uint8)

    # We load the plaintext with known permutation
    P_attack_perm = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][train_begin:train_end], dtype=np.uint8)
    P_attack = P_attack_perm[np.arange(P_attack_perm.shape[0])[:, None], perm_index]

    # We load the key with known permutation
    K_attack_perm = np.array(in_file['Attack_traces/metadata'][:]['key'][train_begin:train_end], dtype=np.uint8)
    K_attack = K_attack_perm[np.arange(K_attack_perm.shape[0])[:, None], perm_index]

    # We simulate a fix key (the orginal key is random)
    key_init = np.array([77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105], dtype=np.uint8)

    simulated_key = np.repeat(key_init[np.newaxis, :], P_attack.shape[0], axis=0)
    simulated_P_attack = P_attack ^ K_attack ^ simulated_key

    mul_mask = np.array(in_file['Attack_traces/labels'][:]['alpha_mask'][train_begin:train_end], dtype=np.uint8)
    add_mask = np.array(in_file['Attack_traces/labels'][:]['beta_mask'][train_begin:train_end], dtype=np.uint8)

    return X_attack, simulated_P_attack, simulated_key, Y_attack, mul_mask, add_mask, perm_index, P_attack_perm, K_attack_perm


def delete_rows(x, p, k, y, m, a):
    # DELETE THE RUBUISH DATA!!! THE TRACES IS FLAT ON THESE DATA.
    rows_to_delete = np.where(m == 0)[0]
    x = np.delete(x, rows_to_delete, axis=0)
    p = np.delete(p, rows_to_delete, axis=0)
    k = np.delete(k, rows_to_delete, axis=0)
    m = np.delete(m, rows_to_delete, axis=0)
    a = np.delete(a, rows_to_delete, axis=0)
    y = np.delete(y, rows_to_delete, axis=0)
    return x, p, k, y, m, a


X_profiling, simulated_P_profiling, simulated_key_profiling, maskedsbox_profiling, mul_mask_profiling, add_mask_profiling, perm_index_profiling, P_profiling_perm, K_profiling_perm = load_profiling_ascad_v2(
    "/tudelft.net/staff-umbrella/dlsca/Guilherme/", profiling_traces=50000)
X_attack, simulated_P_attack, simulated_key_attack, maskedsbox_attack, mul_mask_attack, add_mask_attack, perm_index_attack, P_attack_perm, K_attack_perm = load_attack_ascad_v2(
    "/tudelft.net/staff-umbrella/dlsca/Guilherme/", attack_traces=50000)

mul_mask_profiling_byte = mul_mask_profiling[:, target_byte]
add_mask_profiling_byte = add_mask_profiling[:, target_byte]
mul_mask_attack_byte = mul_mask_attack[:, target_byte]
add_mask_attack_byte = add_mask_attack[:, target_byte]

X_profiling, simulated_P_profiling, simulated_key_profiling, maskedsbox_profiling, mul_mask_profiling_byte, add_mask_profiling_byte = delete_rows(
    X_profiling, simulated_P_profiling, simulated_key_profiling, maskedsbox_profiling, mul_mask_profiling_byte, add_mask_profiling_byte)

X_attack, simulated_P_attack, simulated_key_attack, maskedsbox_attack, mul_mask_attack_byte, add_mask_attack_byte = delete_rows(
    X_attack, simulated_P_attack, simulated_key_attack, maskedsbox_attack, mul_mask_attack_byte, add_mask_attack_byte)

print(X_profiling.shape)
print(simulated_P_profiling.shape)
print(simulated_key_profiling.shape)
print(maskedsbox_profiling.shape)
print(mul_mask_profiling.shape)
print(add_mask_profiling.shape)

print(X_attack.shape)
print(simulated_P_attack.shape)
print(simulated_key_attack.shape)
print(maskedsbox_attack.shape)
print(mul_mask_attack.shape)
print(add_mask_attack.shape)

snr_reference_features_share_1 = snr_fast(X_profiling, mul_mask_profiling_byte[:])
snr_reference_features_share_2 = snr_fast(X_profiling, add_mask_profiling_byte[:])
snr_reference_features_share_3 = snr_fast(X_profiling, maskedsbox_profiling[:, 0])

plt.subplot(3, 1, 1)
plt.plot(snr_reference_features_share_1)
plt.subplot(3, 1, 2)
plt.plot(snr_reference_features_share_2)
plt.subplot(3, 1, 3)
plt.plot(snr_reference_features_share_3)
plt.savefig("snr_reference_features_profiling.png")
plt.close()

snr_reference_features_share_1 = snr_fast(X_attack, mul_mask_attack_byte[:])
snr_reference_features_share_2 = snr_fast(X_attack, add_mask_attack_byte[:])
snr_reference_features_share_3 = snr_fast(X_attack, maskedsbox_attack[:, 0])

plt.subplot(3, 1, 1)
plt.plot(snr_reference_features_share_1)
plt.subplot(3, 1, 2)
plt.plot(snr_reference_features_share_2)
plt.subplot(3, 1, 3)
plt.plot(snr_reference_features_share_3)
plt.savefig("snr_reference_features_attack.png")
plt.close()

target_byte_index = 0

n_profiling = len(X_profiling)
n_attack = len(X_attack)

n_sim = 100
p_s1 = 25
p_s2 = 50
p_s3 = 75

x_prof = np.zeros((n_profiling, n_sim))
x_attack = np.zeros((n_profiling, n_sim))

for i in tqdm(range(n_profiling)):
    new_trace = np.random.randint(0, 256, n_sim) + np.random.normal(0, 0.1, n_sim)

    # first share - multiplicative mask alpha
    new_trace[p_s1] = mul_mask_profiling_byte[i] + np.random.normal(0, 0.1, 1)
    for j in range(15):
        new_trace[p_s1 + j] = mul_mask_profiling_byte[i] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_s1 - j] = mul_mask_profiling_byte[i] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)

    # second share - additive mask beta
    new_trace[p_s2] = add_mask_profiling_byte[i] + np.random.normal(0, 0.1, 1)
    for j in range(15):
        new_trace[p_s2 + j] = add_mask_profiling_byte[i] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_s2 - j] = add_mask_profiling_byte[i] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)

    # third share - masked sbox out GF256(alpha, S(k ^ p)) ^ m
    new_trace[p_s3] = maskedsbox_profiling[i, target_byte_index] + np.random.normal(0, 0.1, 1)
    for j in range(15):
        new_trace[p_s3 + j] = maskedsbox_profiling[i, target_byte_index] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_s3 - j] = maskedsbox_profiling[i, target_byte_index] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)

    x_prof[i] = new_trace

for i in tqdm(range(n_attack)):
    new_trace = np.random.randint(0, 256, n_sim) + np.random.normal(0, 0.1, n_sim)

    # first share - multiplicative mask alpha
    new_trace[p_s1] = mul_mask_attack_byte[i] + np.random.normal(0, 0.1, 1)
    for j in range(15):
        new_trace[p_s1 + j] = mul_mask_attack_byte[i] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_s1 - j] = mul_mask_attack_byte[i] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)

    # second share - additive mask beta
    new_trace[p_s2] = add_mask_attack_byte[i] + np.random.normal(0, 0.1, 1)
    for j in range(15):
        new_trace[p_s2 + j] = add_mask_attack_byte[i] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_s2 - j] = add_mask_attack_byte[i] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)

    # third share - masked sbox out GF256(alpha, S(k ^ p)) ^ m
    new_trace[p_s3] = maskedsbox_attack[i, target_byte_index] + np.random.normal(0, 0.1, 1)
    for j in range(15):
        new_trace[p_s3 + j] = maskedsbox_attack[i, target_byte_index] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_s3 - j] = maskedsbox_attack[i, target_byte_index] + np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)

    x_attack[i] = new_trace

snr_reference_features_share_1 = snr_fast(x_prof, mul_mask_profiling_byte[:])
snr_reference_features_share_2 = snr_fast(x_prof, add_mask_profiling_byte[:])
snr_reference_features_share_3 = snr_fast(x_prof, maskedsbox_profiling[:, 0])

plt.plot(snr_reference_features_share_1)
plt.plot(snr_reference_features_share_2)
plt.plot(snr_reference_features_share_3)
plt.savefig("snr_reference_features_profiling_sim.png")
plt.close()

snr_reference_features_share_1 = snr_fast(x_attack, mul_mask_attack_byte[:])
snr_reference_features_share_2 = snr_fast(x_attack, add_mask_attack_byte[:])
snr_reference_features_share_3 = snr_fast(x_attack, maskedsbox_attack[:, 0])

plt.plot(snr_reference_features_share_1)
plt.plot(snr_reference_features_share_2)
plt.plot(snr_reference_features_share_3)
plt.savefig("snr_reference_features_attack_sim.png")
plt.close()

out_file = h5py.File('/tudelft.net/staff-umbrella/dlsca/Guilherme/aes_sim_mask_reference_v2.h5', 'w')

profiling_index = [n for n in range(n_profiling)]
attack_index = [n for n in range(n_attack)]

# traces profiling
profiling_traces_group = out_file.create_group("Profiling_traces")
profiling_traces_group.create_dataset(name="traces", data=x_prof, dtype=x_prof.dtype)
# traces attack
attack_traces_group = out_file.create_group("Attack_traces")
attack_traces_group.create_dataset(name="traces", data=x_attack, dtype=x_attack.dtype)

# metadata profiling
metadata_type_profiling = np.dtype([("plaintext", P_profiling_perm.dtype, (len(P_profiling_perm[0]),)),
                                    ("key", K_profiling_perm.dtype, (len(K_profiling_perm[0]),)), ])
profiling_metadata = np.array([(P_profiling_perm[n], K_profiling_perm[n]) for n in profiling_index], dtype=metadata_type_profiling)
profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)
# metadata attack
metadata_type_attack = np.dtype([("plaintext", P_attack_perm.dtype, (len(P_attack_perm[0]),)),
                                 ("key", K_attack_perm.dtype, (len(K_attack_perm[0]),)), ])
attack_metadata = np.array([(P_attack_perm[n], K_attack_perm[n]) for n in attack_index], dtype=metadata_type_attack)
attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

# labels profiling
labels_type_profiling = np.dtype([("sbox_masked", maskedsbox_profiling.dtype, (len(maskedsbox_profiling[0]),)),
                                  ("perm_index", perm_index_profiling.dtype, (len(perm_index_profiling[0]),)),
                                  ("alpha_mask", mul_mask_profiling.dtype, (len(mul_mask_profiling[0]),)),
                                  ("beta_mask", add_mask_profiling.dtype, (len(add_mask_profiling[0]),)),
                                  ])
profiling_labels = np.array(
    [(maskedsbox_profiling[n], perm_index_profiling[n], mul_mask_profiling[n], add_mask_profiling[n]) for n in profiling_index],
    dtype=labels_type_profiling)
profiling_traces_group.create_dataset("labels", data=profiling_labels, dtype=labels_type_profiling)
# labels attack
labels_type_attack = np.dtype([("sbox_masked", maskedsbox_attack.dtype, (len(maskedsbox_attack[0]),)),
                               ("perm_index", perm_index_attack.dtype, (len(perm_index_attack[0]),)),
                               ("alpha_mask", mul_mask_attack.dtype, (len(mul_mask_attack[0]),)),
                               ("beta_mask", add_mask_attack.dtype, (len(add_mask_attack[0]),)),
                               ])
attack_labels = np.array([(maskedsbox_attack[n], perm_index_attack[n], mul_mask_attack[n], add_mask_attack[n]) for n in attack_index],
                         dtype=labels_type_attack)
attack_traces_group.create_dataset("labels", data=attack_labels, dtype=labels_type_attack)

out_file.flush()
out_file.close()
