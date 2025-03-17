import h5py
import numpy as np
from tqdm import tqdm
import random

aes_sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

filepath = "/tudelft.net/staff-umbrella/dlsca/Guilherme/ascad-variable.h5"

aes_key = "00112233445566778899AABBCCDDEEFF"
target_byte_index = 2

n_profiling = 200000
n_attack = 100000

in_file = h5py.File(filepath, "r")
profiling_plaintexts = in_file['Profiling_traces/metadata']['plaintext']
profiling_keys = in_file['Profiling_traces/metadata']['key']
profiling_masks = in_file['Profiling_traces/metadata']['masks']

attack_plaintexts = in_file['Profiling_traces/metadata']['plaintext']
attack_keys = in_file['Profiling_traces/metadata']['key']
attack_masks = in_file['Profiling_traces/metadata']['masks']

n_sim = 100
p_m = 25
p_s = 75

x_prof = np.zeros((n_profiling, n_sim))
x_attack = np.zeros((n_profiling, n_sim))

for i in tqdm(range(n_profiling)):
    new_trace = np.random.randint(0, 256, n_sim) + np.random.normal(0, 0.1, n_sim)
    # first share - mask m
    new_trace[p_m] = profiling_masks[i, 2]
    new_trace[p_m] += np.random.normal(0, 0.1, 1)
    for j in range(25):
        new_trace[p_m + j] = profiling_masks[i, 2]
        new_trace[p_m + j] += np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_m - j] = profiling_masks[i, 2]
        new_trace[p_m - j] += np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
    # second share - masked sbox out S(k ^ p) ^ m
    new_trace[p_s] = aes_sbox[profiling_keys[i, target_byte_index] ^ profiling_plaintexts[i, target_byte_index]] ^ profiling_masks[
        i, target_byte_index]
    new_trace[p_s] += np.random.normal(0, 0.1, 1)
    for j in range(25):
        new_trace[p_s + j] = aes_sbox[profiling_keys[i, target_byte_index] ^ profiling_plaintexts[i, target_byte_index]] ^ profiling_masks[
            i, target_byte_index]
        new_trace[p_s + j] += np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_s - j] = aes_sbox[profiling_keys[i, target_byte_index] ^ profiling_plaintexts[i, target_byte_index]] ^ profiling_masks[
            i, target_byte_index]
        new_trace[p_s - j] += np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
    x_prof[i] = new_trace

for i in tqdm(range(n_attack)):
    new_trace = np.random.randint(0, 256, n_sim) + np.random.normal(0, 0.1, n_sim)
    # first share - mask m
    new_trace[p_m] = attack_masks[i, 2] + np.random.normal(0, 0.1, 1)
    new_trace[p_m] += np.random.normal(0, 0.1, 1)
    for j in range(25):
        new_trace[p_m + j] = attack_masks[i, 2]
        new_trace[p_m + j] += np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_m - j] = attack_masks[i, 2]
        new_trace[p_m - j] += np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
    # second share - masked sbox out S(k ^ p) ^ m
    new_trace[p_s] = aes_sbox[attack_keys[i, target_byte_index] ^ attack_plaintexts[i, target_byte_index]] ^ attack_masks[
        i, target_byte_index]
    new_trace[p_s] += np.random.normal(0, 0.1, 1)
    for j in range(25):
        new_trace[p_s + j] = aes_sbox[attack_keys[i, target_byte_index] ^ attack_plaintexts[i, target_byte_index]] ^ attack_masks[
            i, target_byte_index]
        new_trace[p_s + j] += np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
        new_trace[p_s - j] = aes_sbox[attack_keys[i, target_byte_index] ^ attack_plaintexts[i, target_byte_index]] ^ attack_masks[
            i, target_byte_index]
        new_trace[p_s - j] += np.random.normal(0, 0.1 + 0.005 * random.randint(0, 25), 1)
    x_attack[i] = new_trace

out_file = h5py.File('/tudelft.net/staff-umbrella/dlsca/Guilherme/aes_sim_mask_reference.h5', 'w')

profiling_index = [n for n in range(n_profiling)]
attack_index = [n for n in range(n_attack)]

profiling_traces_group = out_file.create_group("Profiling_traces")
attack_traces_group = out_file.create_group("Attack_traces")

profiling_traces_group.create_dataset(name="traces", data=x_prof, dtype=x_prof.dtype)
attack_traces_group.create_dataset(name="traces", data=x_attack, dtype=x_attack.dtype)

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
