import numpy as np
import h5py
from tensorflow.keras.utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

""" AES128 variables """
shift_row_mask = np.array([0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11])

s_box = np.array([
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
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
])

inv_s_box = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
])

r_con = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


class ReadAESHDMM:

    def __init__(self, n_profiling, n_validation, n_attack, target_byte, leakage_model, file_path, first_sample=0, number_of_samples=3250):
        self.name = "aes_hd_mm"
        self.n_profiling = n_profiling
        self.n_validation = n_validation
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.fs = first_sample
        self.ns = number_of_samples
        self.classes = 9 if leakage_model == "HW" else 256

        self.x_profiling = []
        self.x_validation = []
        self.x_attack = []

        self.y_profiling = []
        self.y_validation = []
        self.y_attack = []

        self.profiling_labels = []
        self.validation_labels = []
        self.attack_labels = []

        self.labels_key_hypothesis_validation = None
        self.labels_key_hypothesis_attack = None
        self.share1_profiling, self.share2_profiling, self.share1_attack, self.share2_attack = None, None, None, None

        self.round_key = self.get_last_round_key("000102030405060708090A0B0C0D0E0F")
        self.correct_key_validation = self.round_key[target_byte]
        self.correct_key_attack = self.round_key[target_byte]

        self.profiling_keys = None
        self.profiling_plaintexts = None
        self.profiling_masks = None
        self.attack_plaintexts = None
        self.attack_masks = None
        self.attack_keys = None

        self.load_dataset()

    def load_dataset(self):
        in_file = h5py.File(self.file_path, "r")

        profiling_samples = np.array(in_file['Profiling_traces/traces'][:self.n_profiling])
        attack_samples = np.array(in_file['Attack_traces/traces'][:self.n_attack + self.n_validation])
        profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
        profiling_ciphertext = in_file['Profiling_traces/metadata']['ciphertext']
        profiling_key = in_file['Profiling_traces/metadata']['key']
        profiling_mask = in_file['Profiling_traces/metadata']['masks']

        attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
        attack_ciphertext = in_file['Attack_traces/metadata']['ciphertext']
        attack_key = in_file['Attack_traces/metadata']['key']
        attack_mask = in_file['Attack_traces/metadata']['masks']

        profiling_plaintexts = profiling_plaintext[:self.n_profiling]
        profiling_ciphertexts = profiling_ciphertext[:self.n_profiling]
        profiling_keys = profiling_key[:self.n_profiling]
        profiling_masks = profiling_mask[:self.n_profiling]
        validation_plaintexts = attack_plaintext[:self.n_validation]
        validation_ciphertexts = attack_ciphertext[:self.n_validation]
        validation_keys = attack_key[:self.n_validation]
        validation_masks = attack_mask[:self.n_validation]
        attack_plaintexts = attack_plaintext[self.n_validation:self.n_validation + self.n_attack]
        attack_ciphertexts = attack_ciphertext[self.n_validation:self.n_validation + self.n_attack]
        attack_keys = attack_key[self.n_validation:self.n_validation + self.n_attack]
        attack_masks = attack_mask[self.n_validation:self.n_validation + self.n_attack]

        self.profiling_keys = profiling_keys
        self.profiling_plaintexts = profiling_plaintexts
        self.profiling_masks = profiling_masks
        self.attack_keys = attack_keys
        self.attack_plaintexts = attack_plaintexts
        self.attack_masks = attack_masks

        self.x_profiling = profiling_samples[:, self.fs:self.fs + self.ns]
        self.x_validation = attack_samples[:self.n_validation, self.fs:self.fs + self.ns]
        self.x_attack = attack_samples[self.n_validation:self.n_validation + self.n_attack, self.fs:self.fs + self.ns]

        self.profiling_labels = self.aes_labelize(profiling_ciphertexts, profiling_keys)
        self.validation_labels = self.aes_labelize(validation_ciphertexts, validation_keys)
        self.attack_labels = self.aes_labelize(attack_ciphertexts, attack_keys)

        self.y_profiling = to_categorical(self.profiling_labels, num_classes=self.classes)
        self.y_validation = to_categorical(self.validation_labels, num_classes=self.classes)
        self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)

        self.labels_key_hypothesis_validation = self.create_labels_key_guess(validation_ciphertexts)
        self.labels_key_hypothesis_attack = self.create_labels_key_guess(attack_ciphertexts)

        self.share1_profiling, self.share2_profiling, self.share1_attack, self.share2_attack = self.create_intermediates(
            profiling_ciphertexts,
            profiling_masks,
            profiling_key,
            attack_ciphertexts,
            attack_keys,
            attack_masks,
            self.n_profiling,
            self.n_attack
        )

    def rescale(self, reshape_to_cnn, scaler_type="standard_scaler"):
        self.x_profiling = np.array(self.x_profiling)
        self.x_validation = np.array(self.x_validation)
        self.x_attack = np.array(self.x_attack)

        if scaler_type == "horizontal_min_max":
            self.x_profiling = MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.x_profiling.T).T
            if self.n_validation > 0:
                self.x_validation = MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.x_validation.T).T
            self.x_attack = MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.x_attack.T).T
        else:
            scaler = StandardScaler()
            self.x_profiling = scaler.fit_transform(self.x_profiling)
            if self.n_validation > 0:
                self.x_validation = scaler.transform(self.x_validation)
            self.x_attack = scaler.transform(self.x_attack)

        if reshape_to_cnn:
            print("reshaping to 3 dims")
            self.x_profiling = self.x_profiling.reshape((self.x_profiling.shape[0], self.x_profiling.shape[1], 1))
            self.x_validation = self.x_validation.reshape((self.x_validation.shape[0], self.x_validation.shape[1], 1))
            self.x_attack = self.x_attack.reshape((self.x_attack.shape[0], self.x_attack.shape[1], 1))

    def xor_bytes(self, a, b):
        return [a[i] ^ b[i] for i in range(len(a))]

    def expand_key(self, master_key):
        iteration_count = 0
        for i in range(4, 44):
            word = list(master_key[len(master_key) - 4:])

            if i % 4 == 0:
                word.append(word.pop(0))
                word = [s_box[b] for b in word]
                word[0] ^= r_con[i // 4]

            word = self.xor_bytes(word, master_key[iteration_count * 4:iteration_count * 4 + 4])
            for w in word:
                master_key.append(w)

            iteration_count += 1

        return [master_key[16 * i: 16 * (i + 1)] for i in range(len(master_key) // 16)]

    def get_last_round_key(self, key):
        return self.expand_key(bytearray.fromhex(key))[10]

    def get_round_keys(self, keys):
        """ Compute round keys for all keys """

        keys = np.array(keys, dtype='uint8')
        if np.all(keys == keys[0]):
            """ If all keys are equal, then compute round keys for one key only """
            round_keys = self.expand_key(list(keys[0]))
            return np.full([len(keys), len(round_keys), len(round_keys[0])], round_keys)
        else:
            return [self.expand_key(list(key)) for key in keys]

    def aes_labelize(self, ciphertexts, keys):
        """
        Labels for Hammind Distance leakage model: HD(InvSbox(c[i] xor k[i]), c[j]) = InvSbox(c[i] xor k[i]) xor c[j]
        k[i] = target key byte i of round key 10
        c[i] = ciphertext i
        c[j] = ciphertext j (j is different from i because of ShiftRows)
        """

        # print(f"S^-1(c_{byte} xor k_{byte}) xor c_{shift_row_mask[byte]}")

        if len(ciphertexts) == 0:
            return []

        if np.array(keys).ndim == 1:
            """ repeat key if argument keys is a single key candidate (for GE and SR computations)"""
            keys = np.full([len(ciphertexts), 16], keys)

        """ Compute round keys """
        round_keys = self.get_round_keys(keys)
        """ get ciphertext bytes c[i] and c[j]"""
        c_j = [cj[shift_row_mask[self.target_byte]] for cj in ciphertexts]
        c_i = [ci[self.target_byte] for ci in ciphertexts]
        """ get key byte from round key 10 """
        k_i = [ki[10][self.target_byte] for ki in round_keys]
        if self.leakage_model == "HW":
            return np.array([bin(inv_s_box[int(ci) ^ int(ki)] ^ int(cj)).count("1") for ci, cj, ki in
                             zip(np.asarray(c_i[:]), np.asarray(c_j[:]), np.asarray(k_i[:]))])
        else:
            return np.array([inv_s_box[int(ci) ^ int(ki)] ^ int(cj) for ci, cj, ki in
                             zip(np.asarray(c_i[:]), np.asarray(c_j[:]), np.asarray(k_i[:]))])

    def create_labels_key_guess(self, ciphertexts):
        labels_key_hypothesis = np.zeros((256, len(ciphertexts)), dtype='int64')
        for key_byte_hypothesis in range(256):
            key_h = self.round_key.copy()
            key_h[self.target_byte] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis] = self.aes_labelize(ciphertexts, key_h)
        return labels_key_hypothesis

    def create_intermediates(self, profiling_ciphertexts, profiling_masks, profiling_keys, attack_ciphertexts, attack_keys, attack_masks,
                             n_p,
                             n_a):
        share1_profiling = np.zeros((16, n_p))
        share2_profiling = np.zeros((16, n_p))
        profiling_labels = np.zeros((16, n_p))
        share1_attack = np.zeros((16, n_a))
        share2_attack = np.zeros((16, n_a))
        attack_labels = np.zeros((16, n_a))

        profiling_masks = profiling_masks[:n_p]
        attack_masks = attack_masks[:n_a]

        for byte in range(16):

            profiling_labels[byte, :] = self.aes_labelize(profiling_ciphertexts[:n_p], profiling_keys[:n_p])
            attack_labels[byte, :] = self.aes_labelize(attack_ciphertexts[:n_a], attack_keys[:n_a])

            if self.leakage_model == "HW":
                share1_profiling[byte, :] = [bin(int(m)).count("1") for m in profiling_masks[:, byte]]
                share2_profiling[byte, :] = [bin(int(s) ^ int(m)).count("1") for s, m in
                                             zip(profiling_labels[byte, :], profiling_masks[:, byte])]
                share1_attack[byte, :] = [bin(int(m)).count("1") for m in attack_masks[:, byte]]
                share2_attack[byte, :] = [bin(int(s) ^ int(m)).count("1") for s, m in
                                          zip(attack_labels[byte, :], attack_masks[:, byte])]
            else:
                share1_profiling[byte, :] = profiling_masks[:, byte]
                share2_profiling[byte, :] = [int(s) ^ int(m) for s, m in zip(profiling_labels[byte, :], profiling_masks[:, byte])]
                share1_attack[byte, :] = attack_masks[:, byte]
                share2_attack[byte, :] = [int(s) ^ int(m) for s, m in zip(attack_labels[byte, :], attack_masks[:, byte])]

        return share1_profiling, share2_profiling, share1_attack, share2_attack
