import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

def show_sr_result(model, lr_path, hr_path, device):
    model.eval()

    lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(lr_tensor).clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)


    lr_vis = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(lr_vis, 0, 1))
    plt.title("LR (upsampled for viewing)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(sr, 0, 1))
    plt.title("SR (Predicted)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(hr_img, 0, 1))
    plt.title("HR (Ground Truth)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
