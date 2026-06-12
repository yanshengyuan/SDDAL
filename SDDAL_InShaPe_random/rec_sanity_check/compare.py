import os
import numpy as np
import matplotlib.pyplot as plt

root_dir = "./"   # directory with .npy files
out_dir = "diff"

os.makedirs(out_dir, exist_ok=True)

files = [f for f in os.listdir(root_dir) if f.endswith(".npy")]
files_set = set(files)

pairs = []

for f in files:
    if f.endswith("_.npy"):
        base = f[:-5] + ".npy"   # remove "_" before .npy
        if base in files_set:
            pairs.append((base, f))

if not pairs:
    print("No matching file pairs found.")
    exit()

print(f"Found {len(pairs)} file pairs\n")

for f1, f2 in sorted(pairs):
    a = np.load(os.path.join(root_dir, f1))
    b = np.load(os.path.join(root_dir, f2))

    if a.shape != b.shape:
        print(f"[SKIP] Shape mismatch: {f1} vs {f2}")
        continue

    abs_res = np.abs(a - b)

    # ---- stats ----
    mean_err = abs_res.mean()
    max_err  = abs_res.max()
    rms_err  = np.sqrt(np.mean(abs_res**2))

    print(f"{f1}  <->  {f2}")
    print(f"  mean |res| : {mean_err:.6e}")
    print(f"  max  |res| : {max_err:.6e}")
    print(f"  RMS  |res| : {rms_err:.6e}")

    # ---- draw error map ----
    vmax = np.percentile(abs_res, 99)  # avoid outliers dominating
    plt.figure(figsize=(5, 5))
    im = plt.imshow(abs_res, cmap="viridis", vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"|{f1} - {f2}|", fontsize=9)
    plt.axis("off")

    save_name = f1.replace(".npy", "_abs_error.png")
    plt.savefig(os.path.join(out_dir, save_name),
                dpi=200, bbox_inches="tight")
    plt.close()
