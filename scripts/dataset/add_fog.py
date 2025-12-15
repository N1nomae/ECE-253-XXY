import torch
import cv2
import os
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


img_path = Path(r"E:\ECE_253\fog\images")
depth_path = Path(r"E:\ECE_253\fog\fog")
hazy_path = Path(r"E:\ECE_253\fog\foggy")

# Generated Fog Strength, randomly chosen between min and max
fog_strength_min = 0.65
fog_strength_max = 0.85

# Fog color [R, G, B]
fog_color = np.array([200, 200, 200], dtype=np.uint8)

model_name = "DPT_Hybrid"

depth_path.mkdir(parents=True, exist_ok=True)
hazy_path.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

midas = torch.hub.load("intel-isl/MiDaS", model_name)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

midas.to(device)
midas.eval()

print(f"Using model = {model_name} on device = {device}")


# Generate depth map

imglist = [f for f in os.listdir(img_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

with tqdm(total=len(imglist), desc="Depth Estimation") as pbar:
    for img in imglist:
        full_path = img_path / img
        image = cv2.imread(str(full_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if model_name == "MiDaS":
            transform = midas_transforms.default_transform
        elif model_name == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        input_batch = transform(image).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

        # Normalize depth map
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

        # Save depth map
        depth_uint8 = (depth_map_normalized * 255).astype(np.uint8)
        depth_resized = cv2.resize(depth_uint8, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_LANCZOS4)

        depth_out_file = depth_path / img
        depth_out_file = depth_out_file.with_suffix(".png")
        cv2.imwrite(str(depth_out_file), depth_resized)

        pbar.update(1)


# Generate Fog

with tqdm(total=len(imglist), desc="Fog Effect") as pbar:
    for filename in imglist:
        image_path = img_path / filename
        depth_img_path = depth_path / filename
        depth_img_path = depth_img_path.with_suffix(".png")

        original = cv2.imread(str(image_path))
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        depth_map = cv2.imread(str(depth_img_path), cv2.IMREAD_GRAYSCALE)

        if depth_map is None or original_rgb is None:
            print(f"Skipping {filename}, unable to load image or depth map.")
            pbar.update(1)
            continue

        depth_norm = depth_map.astype(np.float32) / 255.0
        depth_inverted = 1.0 - depth_norm

        # Randomly chosen between min and max
        fog_strength = random.uniform(fog_strength_min, fog_strength_max)
        
        # Fog intensity map
        fog_intensity = depth_inverted * fog_strength
        fog_intensity = np.clip(fog_intensity, 0, 1)

        fog_layer = np.ones_like(original_rgb, dtype=np.float32) * fog_color
        fogged = original_rgb.astype(np.float32) * (1 - fog_intensity[..., None]) + \
                 fog_layer * fog_intensity[..., None]

        fogged_uint8 = np.clip(fogged, 0, 255).astype(np.uint8)

        # Save the foggy pic
        fog_save_path = hazy_path / filename
        fogged_bgr = cv2.cvtColor(fogged_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fog_save_path), fogged_bgr)

        pbar.update(1)

print("Generated foggy images")
