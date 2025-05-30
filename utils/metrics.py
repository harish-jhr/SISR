import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_sr(gt_dir, sr_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))])
    sr_files = sorted([f for f in os.listdir(sr_dir) if f.endswith(('.png', '.jpg'))])

    psnr_scores = []
    ssim_scores = []

    for gt_name, sr_name in zip(gt_files, sr_files):
        gt_path = os.path.join(gt_dir, gt_name)
        sr_path = os.path.join(sr_dir, sr_name)

        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR).resize((480,320))
        sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)

        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)

        if gt_img.shape != sr_img.shape:
            raise ValueError(f"Shape mismatch: {gt_name} vs {sr_name}")

        psnr = peak_signal_noise_ratio(gt_img, sr_img, data_range=255)
        ssim = structural_similarity(gt_img, sr_img, multichannel=True, data_range=255)

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
