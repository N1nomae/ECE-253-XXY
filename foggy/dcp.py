import os
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_DIR = "foggy65_85"
OUTPUT_DIR = "output_dcp"
EXTS = [".jpg", ".jpeg", ".png"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_dark_channel(img, window_size=15):
    """
    img: H x W x 3, float32 in [0,1]
    """
    min_per_pixel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_per_pixel, kernel)
    return dark

def estimate_atmospheric_light(img, dark, top_percent=0.001):
    """
    Improved atmospheric light estimation: more accurate and avoids selecting overexposed pixels
    """
    h, w = dark.shape
    num_pixels = h * w
    num_top = max(int(num_pixels * top_percent), 1)

    dark_vec = dark.reshape(-1)
    img_vec = img.reshape(-1, 3)

    indices = np.argsort(dark_vec)[-num_top:]
    
    candidate_pixels = img_vec[indices]
    brightness = np.sum(candidate_pixels, axis=1)
    brightness_sorted_idx = np.argsort(brightness)
    lower_idx = int(len(brightness_sorted_idx) * 0.75)
    upper_idx = int(len(brightness_sorted_idx) * 0.95)
    valid_candidates = candidate_pixels[brightness_sorted_idx[lower_idx:upper_idx]]
    
    if len(valid_candidates) > 0:
        A = valid_candidates[np.argmax(np.sum(valid_candidates, axis=1))]
    else:
        A = candidate_pixels[np.argmax(brightness)]
    
    A = np.maximum(A, 0.7)
    return A

def guided_filter(I, p, r=60, eps=0.0001):
    """
    Guided filter: refines transmission map to reduce block artifacts
    I: guide image (H x W x 3 or H x W)
    p: input image (H x W)
    r: filter radius
    eps: regularization parameter
    """
    if len(I.shape) == 3:
        I_gray = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        I_gray = I
    
    mean_I = cv2.boxFilter(I_gray, cv2.CV_32F, (r*2+1, r*2+1))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r*2+1, r*2+1))
    mean_Ip = cv2.boxFilter(I_gray * p, cv2.CV_32F, (r*2+1, r*2+1))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I_gray * I_gray, cv2.CV_32F, (r*2+1, r*2+1))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r*2+1, r*2+1))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r*2+1, r*2+1))
    
    q = mean_a * I_gray + mean_b
    return q

def estimate_transmission(img, A, omega=0.95, window_size=15):
    """
    Estimate transmission t(x)
    """
    normed = img / (A + 1e-6)
    dark_normed = get_dark_channel(normed, window_size=window_size)
    t = 1 - omega * dark_normed
    return t

def recover_image(img, A, t, t0=0.15):
    """
    Recover J from I, A, t
    Uses larger t0 to avoid over-enhancement
    """
    t = np.clip(t, t0, 1.0)
    J = (img - A) / t[..., None] + A
    J = np.clip(J, 0, 1)
    return J

def dehaze_dcp(bgr_img):
    """
    Improved DCP dehazing algorithm
    """
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    dark = get_dark_channel(img, window_size=15)
    
    A = estimate_atmospheric_light(img, dark, top_percent=0.001)
    
    t = estimate_transmission(img, A, omega=0.95, window_size=15)
    
    t_refined = guided_filter(img, t, r=60, eps=0.0001)
    t_refined = np.clip(t_refined, 0, 1)
    
    J = recover_image(img, A, t_refined, t0=0.15)
    
    out = (np.clip(J, 0, 1) * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out_bgr

def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in EXTS

def main():
    files = [f for f in os.listdir(INPUT_DIR) if is_image_file(f)]
    print(f"Found {len(files)} images in {INPUT_DIR}")

    for fname in tqdm(files):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)

        img = cv2.imread(in_path)
        if img is None:
            print(f"Warning: failed to read {in_path}")
            continue

        dehazed = dehaze_dcp(img)
        cv2.imwrite(out_path, dehazed)

    print(f"Done! DCP dehazed images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
