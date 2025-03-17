import matplotlib.pyplot as plt
from paper_9_gan.cgan_sca_release.src.datasets.load_ascadf import *


def add_gaussian_noise(ds, std):
    noise = np.random.normal(0, std, np.shape(ds.x_profiling))
    return np.add(ds.x_profiling, noise)


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


target_byte = 2
dataset = ReadASCADf(50000, 0, 10000, 2, "ID", "D:/traces/ASCAD.h5", number_of_samples=700)

max_snr_share_1 = []
max_snr_share_2 = []
std_levels = []
std_base = 0.02
for std_index in range(100):
    ds_noise = add_gaussian_noise(dataset, std_base * std_index)

    snr_val_share_1 = snr_fast(np.array(ds_noise, dtype=np.int16), np.asarray(dataset.share1_profiling[target_byte, :]))
    snr_val_share_2 = snr_fast(np.array(ds_noise, dtype=np.int16), np.asarray(dataset.share2_profiling[target_byte, :]))
    snr_val_share_1[np.isnan(snr_val_share_1)] = 0
    snr_val_share_2[np.isnan(snr_val_share_2)] = 0

    max_snr_share_1.append(np.max(snr_val_share_1))
    max_snr_share_2.append(np.max(snr_val_share_2))
    std_levels.append(std_base * std_index)

    print(f"Std level: {std_base * std_index}")

plt.plot(std_levels, max_snr_share_1, label="Share 1")
plt.plot(std_levels, max_snr_share_2, label="Share 1")
plt.xlabel("Added std Gaussian noise")
plt.ylabel("SNR")
plt.yscale("log")
plt.show()


