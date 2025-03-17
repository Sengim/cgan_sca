import numpy as np
import h5py
from tensorflow.keras.utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

def hw(input: np.uint32):
    out = 0
    temp = input
    for i in range(32):
        if temp % 2 == 1:
            out = out + 1
        temp = temp >> 1
    return out

def hw_8(input: np.uint8):
    out = 0
    temp = input
    for i in range(8):
        if temp % 2 == 1:
            out = out + 1
        temp = temp >> 1
    return out

class ReadSpookSW3:

    def __init__(self, n_profiling, n_validation, n_attack, target_byte, leakage_model, file_path, first_sample=0, number_of_samples=700):
        self.name = "spook_sw3"
        self.n_profiling = n_profiling
        self.n_validation = n_validation
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.fs = first_sample
        self.ns = number_of_samples
        #self.classes = 33
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
        self.share1_profiling, self.share2_profiling, self.share3_profiling, self.share1_validation, self.share2_validation, self.share3_validation  = None, None, None, None, None, None

        #self.round_key = "4DFBE0F27221FE10A78D4ADC8E490469"
        #self.correct_key_validation = bytearray.fromhex(self.round_key)[target_byte]
        #self.correct_key_attack = bytearray.fromhex(self.round_key)[target_byte]

        self.profiling_keys = None
        self.profiling_plaintexts = None
        self.attack_plaintexts = None

        self.load_dataset()

    def load_dataset(self):
        in_file = h5py.File(self.file_path, "r")

        profiling_samples = np.array(in_file['Profiling_traces/traces'][:self.n_profiling+self.n_validation], dtype=np.uint16)
        attack_samples = np.array(in_file['Attack_traces/traces'][:self.n_attack], dtype=np.uint16)
        profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
        attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
        profiling_key = in_file['Profiling_traces/metadata']['key']
        attack_key = in_file['Attack_traces/metadata']['key']
        profiling_mask = in_file['Profiling_traces/metadata']['masks']
        profiling_seed = in_file['Profiling_traces/metadata']['seeds']
        #attack_mask = in_file['Attack_traces/metadata']['masks']

        profiling_plaintexts = profiling_plaintext[:self.n_profiling]
        profiling_keys = profiling_key[:self.n_profiling]
        profiling_masks = profiling_mask[:self.n_profiling]
        profiling_seeds = profiling_seed[:self.n_profiling]
        validation_plaintexts = profiling_plaintext[self.n_profiling:self.n_profiling+self.n_validation]
        validation_keys = profiling_key[self.n_profiling:self.n_profiling+self.n_validation]
        validation_masks = profiling_mask[self.n_profiling:self.n_profiling+self.n_validation]
        validation_seeds = profiling_seed[self.n_profiling:self.n_profiling+self.n_validation]
        attack_plaintexts = attack_plaintext[:self.n_attack]
        self.order = 2
        attack_keys = attack_key[:self.n_attack]
        #attack_masks = attack_mask[:self.n_attack]

        self.profiling_keys = profiling_keys
        self.profiling_plaintexts = profiling_plaintexts
        self.attack_plaintexts = attack_plaintexts

        self.x_profiling = profiling_samples[:self.n_profiling, self.fs:self.fs + self.ns]
        self.x_validation = profiling_samples[self.n_profiling:self.n_validation+self.n_profiling, self.fs:self.fs + self.ns]
        self.x_attack = attack_samples[:self.n_attack, self.fs:self.fs + self.ns]
     
        #print(validation_keys.shape, validation_masks.shape, validation_plaintexts.shape, validation_seeds.shape)
        int_state = in_file['Profiling_traces/metadata']['precomp_int']
        val_state = int_state[self.n_profiling:self.n_validation+self.n_profiling]
        
        #val_state = clyde128_encrypt_masked(validation_plaintexts.T,np.zeros((4,self.n_validation), dtype=np.uint32), validation_masks.T, validation_seeds.T)
        self.val_shares = self.create_shares(val_state)
        self.val_intermediate = self.unmask_shares(self.val_shares)
        #self.val_shares = vec_hw(val_state)


        prof_state = int_state[:self.n_profiling]
        #prof_state = clyde128_encrypt_masked(profiling_plaintexts.T,np.zeros((4,self.n_profiling), dtype=np.uint32), profiling_masks.T, profiling_seeds.T)
        self.prof_shares = self.create_shares(prof_state)
        print(self.prof_shares.shape)
        
        self.prof_intermediate = self.unmask_shares(self.prof_shares)
        
        # self.validation_labels = vec_hw(umask(val_state))[:, self.target_byte]
        # self.profiling_labels = vec_hw(umask(prof_state))[:, self.target_byte]

        self.profiling_labels = np.array([bin(iv).count("1") for iv in self.prof_intermediate[:, self.target_byte]] )if self.leakage_model == "HW" else self.prof_intermediate[:, self.target_byte]
        self.validation_labels =np.array([bin(iv).count("1") for iv in self.val_intermediate[:, self.target_byte]] )if self.leakage_model == "HW" else self.val_intermediate[:, self.target_byte]

        self.y_profiling = to_categorical(self.profiling_labels, num_classes=self.classes)
        self.y_validation = to_categorical(self.validation_labels, num_classes=self.classes)
        #self.y_attack = to_categorical(self.attack_labels, num_classes=self.classes)

        # self.labels_key_hypothesis_validation = self.create_labels_key_guess(validation_plaintexts)
        # self.labels_key_hypothesis_attack = self.create_labels_key_guess(attack_plaintexts)
        # self.share1_profiling, self.share2_profiling, self.share1_attack, self.share2_attack = self.create_shares(profiling_plaintext,
        #                                                                                                           profiling_masks,
        #                                                                                                           profiling_key,
        #                                                                                                           attack_plaintext,
        #                                                                                                           attack_key, attack_masks)
        # self.share1_profiling, self.share2_profiling, self.share1_attack, self.share2_attack = self.create_intermediates(
        #     profiling_plaintext, profiling_masks, profiling_key, attack_plaintexts, attack_keys, attack_masks, self.n_profiling,
        #     self.n_attack)

    def rescale(self, reshape_to_cnn):
        self.x_profiling = np.array(self.x_profiling)
        self.x_validation = np.array(self.x_validation)
        self.x_attack = np.array(self.x_attack)

        scaler = StandardScaler()
        self.x_profiling = scaler.fit_transform(self.x_profiling)
        self.x_validation = scaler.transform(self.x_validation)
        self.x_attack = scaler.transform(self.x_attack)

        if reshape_to_cnn:
            print("reshaping to 3 dims")
            self.x_profiling = self.x_profiling.reshape((self.x_profiling.shape[0], self.x_profiling.shape[1], 1))
            self.x_validation = self.x_validation.reshape((self.x_validation.shape[0], self.x_validation.shape[1], 1))
            self.x_attack = self.x_attack.reshape((self.x_attack.shape[0], self.x_attack.shape[1], 1))

    def create_labels_key_guess(self, plaintexts):
        labels_key_hypothesis = np.zeros((256, len(plaintexts)), dtype='int64')
        for key_byte_hypothesis in range(256):
            key_h = bytearray.fromhex(self.round_key)
            key_h[self.target_byte] = key_byte_hypothesis
            labels_key_hypothesis[key_byte_hypothesis] = self.aes_labelize(plaintexts, key_h)
        return labels_key_hypothesis

    def create_shares(self, states):
        shares = np.zeros((states.shape[0], 16, D), dtype=np.uint8)
        for i in range(16):
            block = i // 4
            shift = ((-i-1) % 4)*8
            #print(i, block, shift)
            for d in range(D):
                #print(block)
                shares[:, i, d] = (states[:, block+ 4*d] >> shift) & 0xff
        return shares
    
    def unmask_shares(self, shares):
        result =  np.zeros((shares.shape[0], 16), dtype=np.uint8)
        for d in range(D):
            result = result ^shares[:, :, d]

        return result


    def create_intermediates(self, profiling_plaintext, profiling_masks, profiling_key, profiling_seed, attack_plaintext, attack_key, attack_masks, attack_seed, n_p,
                             n_a):
        share1_profiling = np.zeros((32, n_p))
        share2_profiling = np.zeros((32, n_p))
        share1_attack = np.zeros((32, n_a))
        share2_attack = np.zeros((32, n_a))

        profiling_masks = profiling_masks[:n_p]
        profiling_key = profiling_key[:n_p]
        profiling_plaintext = profiling_plaintext[:n_p]
        attack_masks = attack_masks[:n_a]
        attack_key = attack_key[:n_a]
        attack_plaintext = attack_plaintext[:n_a]

        for byte in range(16):
            share1_profiling[byte + 2, :] = profiling_masks[:, byte]
            share2_profiling[byte + 2, :] = [aes_sbox[int(p) ^ int(k)] ^ int(m) for p, k, m in
                                             zip(profiling_plaintext[:, byte + 2], profiling_key[:, byte + 2], profiling_masks[:, byte])]
            share1_attack[byte + 2, :] = attack_masks[:, byte]
            share2_attack[byte + 2, :] = [aes_sbox[int(p) ^ int(k)] ^ int(m) for p, k, m in
                                          zip(attack_plaintext[:, byte + 2], attack_key[:, byte + 2], attack_masks[:, byte])]

        share2_profiling[0, :] = [aes_sbox[int(p) ^ int(k)] for p, k, in zip(profiling_plaintext[:, 0], profiling_key[:, 0])]
        share2_attack[0, :] = [aes_sbox[int(p) ^ int(k)] for p, k in zip(attack_plaintext[:, 0], attack_key[:, 0])]
        share2_profiling[1, :] = [aes_sbox[int(p) ^ int(k)] for p, k, in zip(profiling_plaintext[:, 1], profiling_key[:, 1])]
        share2_attack[1, :] = [aes_sbox[int(p) ^ int(k)] for p, k in zip(attack_plaintext[:, 1], attack_key[:, 1])]

        return share1_profiling, share2_profiling, share1_attack, share2_attack

bor = np.bitwise_or
bxor = np.bitwise_xor
band = np.bitwise_and
D = 3


def clyde128_encrypt_masked(state,t,key,seed,Nr=1,step=0):
    """
        This function simulates the behavior of
        multiple executions of Nc masked clyde128

        inputs:
            - state (4,Nc) matrix with each column being a plaintext of an execution
            - t (4,Nc) matrix with each column being a tweak of an execution
            - key ((4*D),Nc)  matrix with each column being a masked key of an execution
            - seed (4,Ns) with each column being the state of the PRNG seed at the beginning of the encryption

            - Nr: number of rounds to simulate
            - step: on what step to stop

        output:
            - (Nc,4*D) where each column is the masked state of the corresponding inputs.

    """
    global seed_tmp
    seed_tmp = seed
    tk = np.array([[t[0],t[1],t[2],t[3]],
            [t[0]^t[2],t[1]^t[3],t[0],t[1]],
            [t[2],t[3],t[0]^t[2],t[1]^t[3]]],dtype=np.uint32)
    masked_state = key.copy()
    XORLS_MASK(masked_state,tk[0])
    XORLS_MASK(masked_state,state)
    off = 0x924
    lfsr = 0x8
    for s in range(0,Nr):
        sbox_layer_masked(masked_state[0:D],
                masked_state[D:D*2],
                masked_state[D*2:D*3],
                masked_state[D*3:D*4],refresh_flag=1)
        if s == (Nr-1) and step == 0:
            return masked_state.T
        lbox_masked(masked_state)
        XORCST_MASK(masked_state,lfsr)
        b = lfsr & 0x1;
        lfsr = (lfsr^(b<<3) | b<<4)>>1;	# update LFSR
        sbox_layer_masked(masked_state[0:D],
                masked_state[D:D*2],
                masked_state[D*2:D*3],
                masked_state[D*3:D*4])
        if s == (Nr-1) and step == 1:
            return masked_state.T
        lbox_masked(masked_state)
        XORCST_MASK(masked_state,lfsr)
        b = lfsr & 0x1;
        lfsr = (lfsr^(b<<3) | b<<4)>>1;	# update LFSR
        off = off>>2

        masked_state ^= key
        XORLS_MASK(masked_state,tk[off&0x3])
    return masked_state.T

###############
### Various operation used by Clyde (see C code for more detailed)
###############
def XORLS_MASK(DEST,OP):
    """
        Performs addition of the unmaksed value OP
        with the shared DEST
    """
    DEST[0,:] ^= OP[0];
    DEST[1*D,:] ^= OP[1];
    DEST[2*D,:] ^= OP[2];
    DEST[3*D,:] ^= OP[3];

def XORCST_MASK(DEST,LFSR):
    """
        Performs round constant addition
        within the masked state
    """
    DEST[0] ^= (LFSR>>3) & 0x1;
    DEST[D] ^= (LFSR>>2) & 0x1;
    DEST[2*D] ^= (LFSR>>1) & 0x1;
    DEST[3*D] ^= (LFSR>>0) & 0x1;

def add_shares(out,a,b):
    """
        Performs addition between two sharing a and b
        and store the result in out.
    """
    for i in range(0,D):
        out[i] = a[i] ^ b[i]

def add_clyde128_masked_state(out,a,b):
    """
        Performs addition of two Clyde states and store
        the result in out.
    """
    for d in range(D):
        for i in range(4):
            j = (i*D)+d
            out[j] = a[j] ^ b[j]

def refresh(shares):
    """
        Performs SNI refresh on the sharing it is implemented
        up to 8 shares
    """
    if D < 4:
        refresh_block_j(shares,1)
    elif D<9:
        refresh_block_j(shares,1)
        refresh_block_j(shares,3)
    else:
        raise Exception("SNI refresh is not implemented for D = %d, max D=8"%(D))

def refresh_block_j(shares,j):
    global seed_tmp
    for i in range(D):
        r,seed_tmp = get_random_tape(seed_tmp,1)
        shares[i] ^= r[0];
        shares[(i+j)%D] ^= r[0]

def mult_shares(out,a,b):
    """
        performs ISW multiplication on two sharings a and b.
        Stores the result in out.
        Takes randomness from the prng state seed_tmp.
    """
    global seed_tmp

    for i in range(D):
        out[i,:] = a[i,:] & b[i,:]

    for i in range(D):
        for j in range(i+1,D):
            rng,seed_tmp = get_random_tape(seed_tmp,1)
            s = rng[0,:]
            tmp = (a[i,:]&b[j,:])^s
            sp = tmp ^ (a[j,:]&b[i,:])
            out[i,:] ^= s
            out[j,:] ^= sp

def sbox_layer_masked(a,b,c,d,refresh_flag=0):
    """
        Applies inplace sbox to the inputs sharings a,b,c,d
        if refresh_flag, a refresh is inserted after the
        first XOR of the Sbox according to Tornado tool.
    """
    y0 = np.zeros(a.shape,dtype=np.uint32)
    y1 = np.zeros(a.shape,dtype=np.uint32)
    y3 = np.zeros(a.shape,dtype=np.uint32)
    tmp = np.zeros(a.shape,dtype=np.uint32)

    mult_shares(tmp,a,b);
    y1[:] = tmp ^ c
    if refresh_flag:
        refresh(y1)
    mult_shares(tmp,d,a);
    y0[:] = tmp ^ b
    mult_shares(tmp,y1,d);
    y3[:] = tmp ^ a
    mult_shares(tmp,y0,y1);
    c[:] = tmp ^ d

    a[:] = y0
    b[:] = y1
    d[:] = y3

def lbox_masked(masked_state):
    """
    Applies lbox to a masked clyde state. Because it is linear, it is a share-wise
    operation.
    """
    for i in range(D):
        masked_state[(0*D) +i], masked_state[(1*D)+i]= lbox(masked_state[(0*D) +i],masked_state[(1*D)+i])
        masked_state[(2*D) +i], masked_state[(3*D)+i]= lbox(masked_state[(2*D) +i],masked_state[(3*D)+i])
def lbox(x, y):
    """
        In place Clyde lbox
    """
    a = x ^ rotr(x, 12)
    b = y ^ rotr(y, 12)
    a = a ^ rotr(a, 3)
    b = b ^ rotr(b, 3)
    a = a ^ rotr(x, 17)
    b = b ^ rotr(y, 17)
    c = a ^ rotr(a, 31)
    d = b ^ rotr(b, 31)
    a = a ^ rotr(d, 26)
    b = b ^ rotr(c, 25)
    a = a ^ rotr(c, 15)
    b = b ^ rotr(d, 15)
    return (a, b)
def rotr(x, c):
    return (x >> c) | ((x << (32-c)) & 0xFFFFFFFF)

def umask(k):
    """"
        unmask N sharing
        N x (4*D) unmask N data in parallel
    """
    out = np.zeros((len(k[:,0]),4),dtype=np.uint32)
    for i in range(4*D):
        out[:,i//D] ^= k[:,i]
    return out

MAX = 16
PRGON = 1
def get_random_tape(seed,l):
    """
        return the randomness from the on-board PRNG
        which is a shadow in sponge mode.

        seed: the actual PRNG state. If a tuple (i,tab,prng_state) , then the PRNG
            is already initialized. If not, the PRNG is initialized.

        l: number of random values to return
    """
    if not type(seed) is tuple:
        # The PRNG is not initialized yet,
        # it is initialized with randomness from the t-function
        seed = np.uint32(seed)
        if isinstance(seed,np.ndarray) and seed.ndim==2:
            rng = np.zeros((l,len(seed[0,:])),dtype=np.uint32)
            prng_state_core = [np.zeros((4,4),dtype=np.uint32) for i in range(len(seed[0,:]))]
            for i,state in enumerate(prng_state_core):
                state[0,:] = seed[:,i]
            prng_tab = np.zeros((MAX,len(seed[0,:])),dtype=np.uint32)
        else:
            rng = np.zeros(l,dtype=np.uint32)
            prng_tab = np.zeros((MAX,1),dtype=np.uint32)
            prng_state_core = [np.zeros((4,4),dtype=np.uint32) for i in range(1)]
            for i,state in enumerate(prng_state_core):
                state[0,:] = seed

        prng_index = MAX
    else:
        # The PRNG is initalized,
        # getting back the actual index and LFSR state
        prng_index,prng_tab,prng_state_core = seed
        rng = np.zeros((l,len(prng_tab[0,:])),dtype=np.uint32)


    #dump l random words
    for i in range(l):
        if prng_index >= MAX:
            fill_table(prng_state_core,prng_tab)
            prng_index = 0
        rng[i] = prng_tab[prng_index]
        prng_index += 1

    if PRGON==0:
        rng[:] = 0
    return rng,(prng_index,prng_tab,prng_state_core)

def fill_table(state_all,tab_all):
    for i in range(0,MAX,8):
        for j,state in enumerate(state_all):
            state_all[j] = shadow(state)
            for n in range(8):
                tab_all[i+n,j] = state_all[j][n//4][n%4]
SMALL_PERM=False
N_STEPS=6

LS_SIZE = 16 # bytes
BLOCK_SIZE =lambda: LS_SIZE if SMALL_PERM else 2*LS_SIZE

RC = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 0, 1, 0),
        (0, 1, 0, 1),
        (1, 1, 1, 0),
        (0, 1, 1, 1),
        ]

def rotr(x, c):
    return (x >> c) | ((x << (32-c)) & 0xFFFFFFFF)

def lbox(x, y):
    a = x ^ rotr(x, 12)
    b = y ^ rotr(y, 12)
    a = a ^ rotr(a, 3)
    b = b ^ rotr(b, 3)
    a = a ^ rotr(x, 17)
    b = b ^ rotr(y, 17)
    c = a ^ rotr(a, 31)
    d = b ^ rotr(b, 31)
    a = a ^ rotr(d, 26)
    b = b ^ rotr(c, 25)
    a = a ^ rotr(c, 15)
    b = b ^ rotr(d, 15)
    return (a, b)

def lbox_inv(x, y):
    a = x ^ rotr(x, 25)
    b = y ^ rotr(y, 25)
    c = x ^ rotr(a, 31)
    d = y ^ rotr(b, 31)
    c = c ^ rotr(a, 20)
    d = d ^ rotr(b, 20)
    a = c ^ rotr(c, 31)
    b = d ^ rotr(d, 31)
    c = c ^ rotr(b, 26)
    d = d ^ rotr(a, 25)
    a = a ^ rotr(c, 17)
    b = b ^ rotr(d, 17)
    a = rotr(a, 16)
    b = rotr(b, 16)
    return (a, b)

def lbox_layer(x):
    return [*lbox(x[0], x[1]), *lbox(x[2], x[3])]

def lbox_layer_inv(x):
    return [*lbox_inv(x[0], x[1]), *lbox_inv(x[2], x[3])]

def sbox_layer(x):
    y1 = (x[0] & x[1]) ^ x[2]
    y0 = (x[3] & x[0]) ^ x[1]
    y3 = (y1 & x[3]) ^ x[0]
    y2 = (y0 & y1) ^ x[3]
    return [y0, y1, y2, y3]

def sbox_layer_inv(x):
    y3 = (x[0] & x[1]) ^ x[2]
    y0 = (x[1] & y3) ^ x[3]
    y1 = (y3 & y0) ^ x[0]
    y2 = (y0 & y1) ^ x[1]
    return [y0, y1, y2, y3]

def add_rc(x, r):
    return list(row ^ RC[r][i] for i, row in enumerate(x))

def xor_states(x, y):
    return list(xr ^ yr for xr, yr in zip(x, y))

def tweakey(key, tweak):
    tx = (tweak[0]^tweak[2], tweak[1]^tweak[3])
    tk = [tweak, (*tx , tweak[0], tweak[1]), (tweak[2], tweak[3], *tx)]
    return [list(k^t for k, t in zip(key, tk_r)) for tk_r in tk]

def xtime(x):
    """Multiplication by polynomial x modulo x^32+x^8+1."""
    b = x >> 31
    return  ((x << 1) & 0xffffffff) ^ b ^ (b << 8)

def dbox(x):
    if SMALL_PERM:
        y = [[0, 0, 0, 0] for _ in range(3)]
        for i in range(4):
            a = x[0][i] ^ x[1][i]
            b = x[0][i] ^ x[2][i]
            c = x[1][i] ^ b
            d = a ^ xtime(b)
            y[0][i] = b ^ d;
            y[1][i] = c;
            y[2][i] = d;
    else:
        y = [[0, 0, 0, 0] for _ in range(4)]
        for i in range(4):
            y[0][i] = x[0][i]
            y[1][i] = x[1][i]
            y[2][i] = x[2][i]
            y[3][i] = x[3][i]
            y[0][i] ^= y[1][i]
            y[2][i] ^= y[3][i]
            y[1][i] ^= y[2][i]
            y[3][i] ^= xtime(y[0][i])
            y[1][i]  = xtime(y[1][i])
            y[0][i] ^= y[1][i]
            y[2][i] ^= xtime(y[3][i])
            y[1][i] ^= y[2][i]
            y[3][i] ^= y[0][i]
    return y

def clyde_encrypt(m, t, k):
    tk = tweakey(k, t)
    x = xor_states(m, tk[0])
    for s in range(N_STEPS):
        for rho in range(2):
            r = 2*s+rho
            x = sbox_layer(x)
            x = lbox_layer(x)
            x = add_rc(x, r)
        x = xor_states(x, tk[(s+1)%3])
    return x

def clyde_decrypt(c, t, k):
    tk = tweakey(k, t)
    x = c
    for s in reversed(range(N_STEPS)):
        x = xor_states(x, tk[(s+1)%3])
        for rho in reversed(range(2)):
            r = 2*s+rho
            x = add_rc(x, r)
            x = lbox_layer_inv(x)
            x = sbox_layer_inv(x)
    x = xor_states(x, tk[0])
    return x

def bytes2state(x):
    return list(x[4*i] | x[4*i+1] << 8 | x[4*i+2] << 16 | x[4*i+3] << 24 for i in range(4))

def state2bytes(x):
    return bytes((r >> 8*i) & 0xFF for r in x for i in range(4))

CST_LFSR_POLY_MASK = 0xc5
CST_LFSR_INIT_VALUE = 0xf8737400
SHADOW_RA_CST_ROW = 1
def update_lfsr(lfsr):
    return ((lfsr << 1) & 0xffffffff) ^ (CST_LFSR_POLY_MASK if lfsr & 0x80000000 else 0)

def shadow(x):
    lfsr = CST_LFSR_INIT_VALUE
    for s in range(N_STEPS):
        x = [lbox_layer(sbox_layer(xi)) for xi in x]
        for xi in x:
            xi[SHADOW_RA_CST_ROW] ^= lfsr
            lfsr = update_lfsr(lfsr)
        x = [sbox_layer(xi) for xi in x]
        x = dbox(x)
        for i in range(len(x[0])):
            x[0][i] ^= lfsr
            lfsr = update_lfsr(lfsr)
    return x

def pad_bytes(b,n=LS_SIZE):
    return b.ljust(n, bytes((0,)))

def init_sponge_state(k, n):
    if len(k) == 32:
        # mu variant
        p = bytearray(k[16:])
        p[-1] &= 0x7F
        p[-1] |= 0x40
        p = bytes2state(p)
    else:
        assert len(k) == 16
        p = (0, 0, 0, 0)
    n = bytes2state(n)
    b = clyde_encrypt(n, p, bytes2state(k))
    if SMALL_PERM:
        x = [b, p, n]
    else:
        x = [b, p, n, [0, 0, 0, 0]]
    return shadow(x)

def compress_block(x, block, mode, nbytes,pad):
    xb = state2bytes(x[0])
    if not SMALL_PERM:
        xb = xb + state2bytes(x[1])
    res = bytes(a ^ b for a, b in zip(xb, block))
    if mode == 'ENC':
        x_bytes = res
    elif mode == 'DEC':
        x_bytes = bytearray(xb)
        for i, b in enumerate(block[:nbytes]):
            x_bytes[i] = b
        if pad:
            x_bytes[nbytes] ^= 0x01
        x_bytes = bytes(x_bytes)
    x[0] = bytes2state(x_bytes[:LS_SIZE])
    if not SMALL_PERM:
        x[1] = bytes2state(x_bytes[LS_SIZE:])
    return x, res[:nbytes]

def compress_data(x, data, mode='ENC'):
    res = b''
    while len(data) >= BLOCK_SIZE():
        x, r = compress_block(x, data[:BLOCK_SIZE()], mode, BLOCK_SIZE(), False)
        res = res+r
        data = data[BLOCK_SIZE():]
        x = shadow(x)
    if data:
        pb = pad_bytes(data + b'\x01', n=BLOCK_SIZE())
        x, r = compress_block(x, pb, mode, len(data), True)
        res = res+r
        x[-2][0] ^= 0x2
        x = shadow(x)
    return x, res

def spook_encrypt(ad, m, k, n):
    x = init_sponge_state(k, n)
    x, _ = compress_data(x, ad)
    if m:
        x[-2][0] ^= 0x1
        x, c = compress_data(x, m)
    else:
        c = b''
    x[1][3] |= 0x80000000
    tag = state2bytes(clyde_encrypt(x[0], x[1], bytes2state(k)))
    return c+tag

def spook_decrypt(ad, c, k, n):
    x = init_sponge_state(k, n)
    x, _ = compress_data(x, ad)
    if len(c) > LS_SIZE:
        x[-2][0] ^= 0x1
        x, m = compress_data(x, c[:-LS_SIZE], mode='DEC')
    else:
        m = b''
    x[1][3] |= 0x80000000
    # NOTE: We do forward tag verification. In leveled implementations against
    # side-channel analysis, inverse tag verification should be performed to
    # enjoy the CIML2 property:
    # cst_time_cmp(x[0], clyde_decrypt(bytes2state(c[-LS_SIZE:], x[1], bytes2state(k))))
    tag = state2bytes(clyde_encrypt(x[0], x[1], bytes2state(k)))
    assert x[0] == clyde_decrypt(bytes2state(tag), x[1], bytes2state(k))
    if tag == c[-LS_SIZE:]:
        return m
    else:
        return None



