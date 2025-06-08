import os
import random
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

def random_circle_mask(size):
    # size = (C, H, W)
    img = np.zeros((size[1], size[2]), np.uint8)
    number = random.randint(16, 64)
    max_h, max_w = size[1], size[2]

    for _ in range(number):
        radius = random.randint(4, min(20, max_h // 4, max_w // 4))
        x, y = random.randint(0, max_h - 1), random.randint(0, max_w - 1)
        cv2.circle(img, (y, x), radius, 255, -1)
    return img

def mask_ratio(mask):
    return np.sum(mask == 255) / mask.size

def adjust_mask_ratio(mask, target_ratio, max_iter=20):
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max_iter):
        current_ratio = mask_ratio(mask)
        diff = current_ratio - target_ratio
        if abs(diff) < 0.01:
            break
        if diff < 0:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def generate_mask_with_ratio(size, target_ratio, max_attempts=100):
    for _ in range(max_attempts):
        mask = random_circle_mask(size)
        mask = adjust_mask_ratio(mask, target_ratio)
        ratio = mask_ratio(mask)
        if abs(ratio - target_ratio) < 0.02:
            return mask
    return mask  # fallback nếu không đạt

def generate_masks_by_ratios(image_dir, output_base, ratios=[0.1, 0.2, 0.3, 0.4, 0.5]):
    os.makedirs(output_base, exist_ok=True)
    image_list = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for ratio in ratios:
        ratio_percent = int(ratio * 100)
        output_dir = os.path.join(output_base, f"mask_{ratio_percent}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating circle masks with ratio ~{ratio_percent}% ...")
        for img_name in tqdm(image_list):
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            size = transforms.ToTensor()(img).size()  # (C, H, W)
            mask = generate_mask_with_ratio(size, ratio)
            save_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".png")
            cv2.imwrite(save_path, mask)

# === RUN ===
if __name__ == "__main__":
    image_dir = "C:/Users/Administrator/CS331/test"
    output_base = "./data_split/mask_test_ratio"
    generate_masks_by_ratios(image_dir, output_base)
