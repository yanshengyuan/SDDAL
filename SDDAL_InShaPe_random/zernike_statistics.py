import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument('--beamshape', type=str)
parser.add_argument('--init_size', type=int)

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# ============================
# User settings
# ============================
args = parser.parse_args()
folder = "Design_" + args.beamshape + "/training_set/zernikes"
initial_size = args.init_size

# ============================
# Read all .npy files
# ============================
all_values = []
files = os.listdir(folder)

for fname in sorted(files, key=natural_key):
    if fname.endswith(".npy"):
        arr = np.load(os.path.join(folder, fname))
        all_values.append(arr)

all_values = all_values[initial_size:]
all_values = np.concatenate(all_values)
print(f"Loaded total coefficients: {len(all_values)}")

# ============================
# Histogram
# ============================
plt.figure()
plt.hist(all_values, bins=50, density=True, alpha=0.6)
plt.xlabel("Zernike coefficient value")
plt.ylabel("Probability density")
plt.title("Histogram of Zernike Coefficients")
plt.grid(True)
plt.tight_layout()
plt.savefig("zernike_histogram.png", dpi=300)
plt.close()

# ============================
# KDE with seaborn
# ============================
plt.figure()
sns.kdeplot(all_values, fill=True, bw_adjust=1.0)
plt.xlabel("Zernike coefficient value")
plt.ylabel("Density")
plt.title("KDE of Zernike Coefficients (sns.kdeplot)")
plt.grid(True)
plt.tight_layout()
plt.savefig("zernike_kde.png", dpi=300)
plt.close()
