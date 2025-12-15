import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
SRC_IMG_DIR = "E:/Learn/ECE253/daytime_yolo/images"
SRC_LABEL_DIR = "E:/Learn/ECE253/daytime_yolo/labels"
OUT_DIR = "E:/Learn/ECE253/daytime_blurred_various"
BLUR_LEVELS = [15]

os.makedirs(os.path.join(OUT_DIR, "images_original"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "labels"), exist_ok=True)
for k in BLUR_LEVELS:
    os.makedirs(os.path.join(OUT_DIR, f"images_blur{k}"), exist_ok=True)

def motion_blur(img, k=25):
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

img_paths = sorted(glob.glob(os.path.join(SRC_IMG_DIR, "*.png")) + 
                   glob.glob(os.path.join(SRC_IMG_DIR, "*.jpg")))

print(f"Found {len(img_paths)} images, processing all...")

for img_path in tqdm(img_paths, desc="Processing images"):
    img_name = os.path.basename(img_path)
    base = os.path.splitext(img_name)[0]

    label_path = os.path.join(SRC_LABEL_DIR, base + ".txt")

    img = cv2.imread(img_path)

    cv2.imwrite(os.path.join(OUT_DIR, "images_original", img_name), img)

    for k in BLUR_LEVELS:
        blurred = motion_blur(img, k=k)
        cv2.imwrite(os.path.join(OUT_DIR, f"images_blur{k}", img_name), blurred)

    out_label_path = os.path.join(OUT_DIR, "labels", base + ".txt")
    if os.path.exists(label_path):
        with open(label_path, "r") as f_src, open(out_label_path, "w") as f_out:
            f_out.write(f_src.read())
    else:
        open(out_label_path, "w").close()
