import h5py
import numpy as np

nt_profiling = 500000
nt_attack = 100000
nt_file = 50000
ns = 3125
fs = 0

directory = "/tudelft.net/staff-umbrella/dlsca/Guilherme"

profiling_trace_folder = ["001_masked_AES", "002_masked_AES", "003_masked_AES", "004_masked_AES", "005_masked_AES", "006_masked_AES",
                          "007_masked_AES", "008_masked_AES", "009_masked_AES", "010_masked_AES"]
profiling_trace_files = ["001_trace_int", "002_trace_int", "003_trace_int", "004_trace_int", "005_trace_int", "006_trace_int",
                         "007_trace_int", "008_trace_int", "009_trace_int", "010_trace_int"]
profiling_trace_plaintexts = ["001_plain", "002_plain", "003_plain", "004_plain", "005_plain", "006_plain",
                              "007_plain", "008_plain", "009_plain", "010_plain"]
profiling_trace_ciphertexts = ["001_cipher", "002_cipher", "003_cipher", "004_cipher", "005_cipher", "006_cipher",
                               "007_cipher", "008_cipher", "009_cipher", "010_cipher"]
profiling_trace_masks = ["001_mask", "002_mask", "003_mask", "004_mask", "005_mask", "006_mask",
                         "007_mask", "008_mask", "009_mask", "010_mask"]

attacking_trace_folder = ["011_masked_AES", "012_masked_AES"]
attacking_trace_files = ["011_trace_int", "012_trace_int"]
attacking_trace_plaintexts = ["011_plain", "012_plain"]
attacking_trace_ciphertexts = ["011_cipher", "012_cipher"]
attacking_trace_masks = ["011_mask", "012_mask"]

profiling_samples = np.zeros((nt_profiling, ns))
profiling_plaintext = np.zeros((nt_profiling, 16))
profiling_ciphertext = np.zeros((nt_profiling, 16))
profiling_masks = np.zeros((nt_profiling, 16))
profiling_key = np.zeros((nt_profiling, 16))

for index, file in enumerate(profiling_trace_files):

    print(file)

    trace_file = open(f"{directory}/aes_hd_mm_files/{profiling_trace_folder[index]}/{file}.txt")
    plaintext_file = open(f"{directory}/aes_hd_mm_files/{profiling_trace_folder[index]}/{profiling_trace_plaintexts[index]}.txt")
    ciphertext_file = open(f"{directory}/aes_hd_mm_files/{profiling_trace_folder[index]}/{profiling_trace_ciphertexts[index]}.txt")
    mask_file = open(f"{directory}/aes_hd_mm_files/{profiling_trace_folder[index]}/{profiling_trace_masks[index]}.txt")

    traces = trace_file.readlines()
    plaintexts = plaintext_file.readlines()
    ciphertexts = ciphertext_file.readlines()
    masks = mask_file.readlines()

    for i, trace in enumerate(traces):
        profiling_samples[i + nt_file * index] = [float(trace_sample) for trace_sample in trace.split(" ")][fs:]
        profiling_plaintext[i + nt_file * index] = [int(trace_plaintext) for trace_plaintext in plaintexts[i].split(" ")]
        profiling_ciphertext[i + nt_file * index] = [int(trace_ciphertext) for trace_ciphertext in ciphertexts[i].split(" ")]
        profiling_masks[i + nt_file * index] = [int(trace_mask) for trace_mask in masks[i].split(" ")]

attack_samples = np.zeros((nt_attack, ns))
attack_plaintext = np.zeros((nt_attack, 16))
attack_ciphertext = np.zeros((nt_attack, 16))
attack_masks = np.zeros((nt_attack, 16))
attack_key = np.zeros((nt_attack, 16))

for index, file in enumerate(attacking_trace_files):

    print(file)

    trace_file = open(f"{directory}/aes_hd_mm_files/{attacking_trace_folder[index]}/{file}.txt")
    plaintext_file = open(f"{directory}/aes_hd_mm_files/{attacking_trace_folder[index]}/{attacking_trace_plaintexts[index]}.txt")
    ciphertext_file = open(f"{directory}/aes_hd_mm_files/{attacking_trace_folder[index]}/{attacking_trace_ciphertexts[index]}.txt")
    mask_file = open(f"{directory}/aes_hd_mm_files/{attacking_trace_folder[index]}/{attacking_trace_masks[index]}.txt")

    traces = trace_file.readlines()
    plaintexts = plaintext_file.readlines()
    ciphertexts = ciphertext_file.readlines()
    masks = mask_file.readlines()

    for i, trace in enumerate(traces):
        attack_samples[i + nt_file * index] = [float(trace_sample) for trace_sample in trace.split(" ")][fs:]
        attack_plaintext[i + nt_file * index] = [int(trace_plaintext) for trace_plaintext in plaintexts[i].split(" ")]
        attack_ciphertext[i + nt_file * index] = [int(trace_ciphertext) for trace_ciphertext in ciphertexts[i].split(" ")]
        attack_masks[i + nt_file * index] = [int(trace_mask) for trace_mask in masks[i].split(" ")]

key = "000102030405060708090A0B0C0D0E0F"
key_int = ([int(x) for x in bytearray.fromhex(key)])

for i in range(nt_profiling):
    for j in range(16):
        profiling_key[i][j] = key_int[j]
for i in range(nt_attack):
    for j in range(16):
        attack_key[i][j] = key_int[j]

profiling_index = [n for n in range(0, nt_profiling)]
attack_index = [n for n in range(0, nt_attack)]

out_file = h5py.File('{}/aes_hd_mm.h5'.format(directory), 'w')

# Create our HDF5 hierarchy in the output file:
# Profiling traces with their labels
# Attack traces with their labels
profiling_traces_group = out_file.create_group("Profiling_traces")
attack_traces_group = out_file.create_group("Attack_traces")
# Datasets in the groups
profiling_traces_group.create_dataset(name="traces", data=profiling_samples, dtype=profiling_samples.dtype)
attack_traces_group.create_dataset(name="traces", data=attack_samples, dtype=attack_samples.dtype)
# Labels in the groups
# profiling_traces_group.create_dataset(name="labels", data=labels_profiling, dtype=labels_profiling.dtype)
# attack_traces_group.create_dataset(name="labels", data=labels_attack, dtype=labels_attack.dtype)
# TODO: deal with the case where "ciphertext" entry is there
# Put the metadata (plaintexts, keys, ...) so that one can check the key rank
metadata_type_profiling = np.dtype([("plaintext", profiling_plaintext.dtype, (len(profiling_plaintext[0]),)),
                                    ("ciphertext", profiling_ciphertext.dtype, (len(profiling_ciphertext[0]),)),
                                    ("masks", profiling_masks.dtype, (len(profiling_masks[0]),)),
                                    ("key", profiling_key.dtype, (len(profiling_key[0]),))
                                    ])
metadata_type_attack = np.dtype([("plaintext", attack_plaintext.dtype, (len(attack_plaintext[0]),)),
                                 ("ciphertext", attack_ciphertext.dtype, (len(attack_ciphertext[0]),)),
                                 ("masks", attack_masks.dtype, (len(attack_masks[0]),)),
                                 ("key", attack_key.dtype, (len(attack_key[0]),))
                                 ])

profiling_metadata = np.array([(profiling_plaintext[n], profiling_ciphertext[n], profiling_masks[n], profiling_key[n]) for n, k in
                               zip(profiling_index, range(0, len(profiling_samples)))], dtype=metadata_type_profiling)
profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

attack_metadata = np.array([(attack_plaintext[n], attack_ciphertext[n], attack_masks[n], attack_key[n]) for n, k in
                            zip(attack_index, range(0, len(attack_samples)))], dtype=metadata_type_attack)
attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

out_file.flush()
out_file.close()

in_file = h5py.File('{}/aes_hd_mm.h5'.format(directory), "r")
profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
profiling_ciphertext = in_file['Profiling_traces/metadata']['ciphertext']
attack_ciphertext = in_file['Attack_traces/metadata']['ciphertext']
profiling_masks = in_file['Profiling_traces/metadata']['masks']
attack_masks = in_file['Attack_traces/metadata']['masks']
profiling_key = in_file['Profiling_traces/metadata']['key']
attack_key = in_file['Attack_traces/metadata']['key']

print(profiling_plaintext)
print(attack_plaintext)
print(profiling_ciphertext)
print(attack_ciphertext)
print(profiling_masks)
print(attack_masks)
print(profiling_key)
print(attack_key)
