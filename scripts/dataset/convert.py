from PIL import Image
import os

# Configuration
MODE = "crop"
TARGET_W, TARGET_H = 1224, 370

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(BASE_DIR, "our_dataset", "night_origin")
output_dir = os.path.join(BASE_DIR, "our_dataset", "new_night_processed")
os.makedirs(output_dir, exist_ok=True)


def resize_and_crop(img):
    """Resize maintaining aspect ratio and center crop to 1224×370"""
    w, h = img.size
    img_aspect = w / h
    target_aspect = TARGET_W / TARGET_H

    # Scale to cover target dimensions
    scale = max(TARGET_W / w, TARGET_H / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - TARGET_W) // 2
    top = (new_h - TARGET_H) // 2
    right = left + TARGET_W
    bottom = top + TARGET_H

    return img_resized.crop((left, top, right, bottom))


for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    input_path = os.path.join(input_dir, filename)

    try:
        img = Image.open(input_path).convert("RGB")
        out_img = resize_and_crop(img)

        output_name = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_name)
        out_img.save(output_path, format="PNG")

        print(f"Processed: {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("All images processed!")
