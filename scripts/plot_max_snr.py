import matplotlib.pyplot as plt
import numpy as np


max_shares = np.load("C:/Users/Sengim/Datasets/model_search_best/ascad-variable_vs_dpa_v42_70000_pca/max_snr_shares.npz")

print(max_shares["max_snr_share_1"])

plt.plot(max_shares["max_snr_share_1"], label="Max SNR share 1")
plt.plot(max_shares["max_snr_share_2"], label="Max SNR share 2")
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("SNR")
plt.legend()
plt.grid(True)
plt.ylim(top=20)
plt.show()