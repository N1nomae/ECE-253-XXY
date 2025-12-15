import os
import shutil

# Auto-detect the script directory (so relative paths work when run from anywhere)
script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()

# Locate an image folder in the current directory
IMG_DIR = None
for folder_name in ["images", "image", "img", "train", "val"]:
    folder_path = os.path.join(script_dir, folder_name)
    if os.path.isdir(folder_path):
        if any(f.lower().endswith((".jpg", ".png", ".jpeg")) for f in os.listdir(folder_path)):
            IMG_DIR = folder_path
            break

if IMG_DIR is None:
    print("Error: No image folder found!")
    exit(1)

# Locate a label folder in the current directory
LABEL_DIR = None
for folder_name in ["labels_yolo", "labels", "label", "annotations"]:
    folder_path = os.path.join(script_dir, folder_name)
    if os.path.isdir(folder_path):
        LABEL_DIR = folder_path
        break

if LABEL_DIR is None:
    print("Warning: No label folder found!")
    LABEL_DIR = os.path.join(script_dir, "labels_yolo")
    os.makedirs(LABEL_DIR, exist_ok=True)

print(f"Image folder: {os.path.basename(IMG_DIR)}")
print(f"Label folder: {os.path.basename(LABEL_DIR)}")

# Collect all image files
image_files = []
for fname in os.listdir(IMG_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        image_files.append(fname)

# Sort by filename for deterministic renaming
image_files.sort()

print(f"Found {len(image_files)} image files")

# Build a rename mapping (old paths -> new paths)
rename_mapping = []
for idx, old_name in enumerate(image_files, start=1):
    # File extension
    _, ext = os.path.splitext(old_name)
    
    # New filenames: 001, 002, 003, ...
    new_img_name = f"{idx:03d}{ext}"
    new_txt_name = f"{idx:03d}.txt"
    
    old_img_path = os.path.join(IMG_DIR, old_name)
    old_txt_path = os.path.join(LABEL_DIR, os.path.splitext(old_name)[0] + ".txt")
    
    new_img_path = os.path.join(IMG_DIR, new_img_name)
    new_txt_path = os.path.join(LABEL_DIR, new_txt_name)
    
    rename_mapping.append({
        'old_img': old_img_path,
        'new_img': new_img_path,
        'old_txt': old_txt_path,
        'new_txt': new_txt_path,
        'old_name': old_name,
        'new_name': new_img_name
    })

# Step 1: rename to temporary names to avoid name collisions
print("\nStep 1: Renaming to temporary names...")
for item in rename_mapping:
    temp_img = item['old_img'] + ".tmp"
    temp_txt = item['old_txt'] + ".tmp" if os.path.exists(item['old_txt']) else None
    
    if os.path.exists(item['old_img']):
        os.rename(item['old_img'], temp_img)
        item['temp_img'] = temp_img
    
    if temp_txt and os.path.exists(item['old_txt']):
        os.rename(item['old_txt'], temp_txt)
        item['temp_txt'] = temp_txt

# Step 2: rename temp files to the final names
print("Step 2: Renaming to final names...")
for item in rename_mapping:
    if 'temp_img' in item:
        os.rename(item['temp_img'], item['new_img'])
        print(f"  {item['old_name']} -> {os.path.basename(item['new_img'])}")
    
    if 'temp_txt' in item and os.path.exists(item['temp_txt']):
        os.rename(item['temp_txt'], item['new_txt'])
        print(f"  {os.path.basename(item['old_txt'])} -> {os.path.basename(item['new_txt'])}")
    elif os.path.exists(item['old_txt']):
        # If we didn't rename to a temp label earlier, rename it directly
        os.rename(item['old_txt'], item['new_txt'])
        print(f"  {os.path.basename(item['old_txt'])} -> {os.path.basename(item['new_txt'])}")

print(f"\nSuccessfully renamed {len(rename_mapping)} files!")

