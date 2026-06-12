import os
import numpy as np
from math import sqrt
from skimage.metrics import structural_similarity as ssim

gt_folder = './Phi_gt/npy'
pred_folder = './Phi_pred/npy'
mae_list = []
ssim_list = []

cnt=0
# Traverse through all files in the gt folder
for gt_filename in os.listdir(gt_folder):
    # Load ground truth and prediction arrays
    gt_path = os.path.join(gt_folder, gt_filename)
    pred_path = os.path.join(pred_folder, gt_filename)

    if os.path.exists(pred_path):
        gt_array = np.load(gt_path)
        pred_array = np.load(pred_path)

        # Calculate RMSE
        diff = np.abs(gt_array - pred_array)
        mae = np.mean(diff)
        mae_list.append(mae)
        
        ssim_value = ssim(gt_array, pred_array, data_range=gt_array.max() - gt_array.min())
        ssim_list.append(ssim_value)

        cnt+=1
        print(cnt)

# Calculate mean RMSE, DSSIM, and FRCM
mean_mae = np.mean(mae_list)
mean_ssim = np.mean(ssim_list)

print(f"Mean MAE: {mean_mae}")
print(f"Mean SSIM: {mean_ssim}")