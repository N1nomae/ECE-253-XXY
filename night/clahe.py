import cv2
import os
from pathlib import Path
import numpy as np

def apply_adaptive_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Adaptive CLAHE enhancement - adjusts parameters based on image brightness
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    if avg_brightness < 50:
        adaptive_clip = clip_limit * 1.5
        adaptive_tile = (6, 6)
    elif avg_brightness < 100:
        adaptive_clip = clip_limit * 1.2
        adaptive_tile = (8, 8)
    else:
        adaptive_clip = clip_limit
        adaptive_tile = tile_grid_size
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=adaptive_tile)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def apply_clahe_enhancement(image_path, output_path, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Apply adaptive CLAHE enhancement to a single image
    """
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to read image: {image_path}")
        return False
    
    enhanced_img = apply_adaptive_clahe(img, clip_limit, tile_grid_size)
    cv2.imwrite(output_path, enhanced_img)
    
    return True

def process_images(input_folder, output_folder, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Batch process all images in a folder
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images")
    print(f"Method: Adaptive CLAHE (auto-adjusts based on brightness)")
    print(f"Base CLAHE params: clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
    print("Processing...\n")
    
    success_count = 0
    for i, image_path in enumerate(image_files, 1):
        output_path = os.path.join(output_folder, image_path.name)
        
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}...", end=" ")
        
        if apply_clahe_enhancement(str(image_path), output_path, clip_limit, tile_grid_size):
            print("Done")
            success_count += 1
        else:
            print("Failed")
    
    print(f"\nProcessing complete! Success: {success_count}/{len(image_files)}")
    print(f"Enhanced images saved to: {output_folder}")

if __name__ == "__main__":
    # Configuration
    input_folder = "images"
    output_folder = "output_CLAHE"
    
    clip_limit = 4.0
    tile_grid_size = (8, 8)
    
    process_images(input_folder, output_folder, clip_limit, tile_grid_size)
    
    print("\n" + "="*60)
    print("Adaptive CLAHE Info:")
    print("- Very dark images (<50): clip_limit * 1.5, tile=(6,6)")
    print("- Dark images (<100): clip_limit * 1.2, tile=(8,8)")
    print("- Normal/bright images: use base parameters")
    print("="*60)

