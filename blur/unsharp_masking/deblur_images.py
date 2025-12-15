"""
Unsharp Masking Image Deblurring Function

This module implements unsharp masking to deblur/sharp images from a dataset.
Formula: Î = λ(I - F_L(I)) + I
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional
from scipy.ndimage import gaussian_filter


def unsharp_mask(image: np.ndarray, 
                 sigma: float = 1.0, 
                 amount: float = 1.0) -> np.ndarray:
    """
    Apply unsharp masking to an image.
    
    Formula: Î = λ(I - F_L(I)) + I
    
    Args:
        image: Input image as numpy array (grayscale or color)
        sigma: Standard deviation for Gaussian low-pass filter (F_L)
        amount: Lambda (λ) - amount coefficient controlling enhancement volume
    
    Returns:
        Enhanced/sharpened image as numpy array
    """
    # Handle color images properly - apply filter to each channel separately
    if len(image.shape) == 3:
        # Color image (BGR format from OpenCV)
        blurred = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[2]):
            blurred[:, :, i] = gaussian_filter(image[:, :, i].astype(np.float32), sigma=sigma)
    else:
        # Grayscale image
        blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)
    
    # Convert original to float32 for calculations
    image_float = image.astype(np.float32)
    
    # Compute difference: I - F_L(I) (edge information)
    edge_info = image_float - blurred
    
    # Apply formula: Î = λ(I - F_L(I)) + I
    enhanced = amount * edge_info + image_float
    
    # Clip values to valid range [0, 255] and convert back to uint8
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def deblur_image_from_path(input_path: Union[str, Path], 
                           output_path: Union[str, Path],
                           sigma: float = 1.0,
                           amount: float = 1.0) -> bool:
    """
    Deblur a single image file using unsharp masking.
    
    Args:
        input_path: Path to input image
        output_path: Path to save deblurred image
        sigma: Standard deviation for Gaussian filter
        amount: Enhancement amount coefficient
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: Could not read image from {input_path}")
            return False
        
        # Apply unsharp masking
        deblurred = unsharp_mask(image, sigma=sigma, amount=amount)
        
        # Save deblurred image
        cv2.imwrite(str(output_path), deblurred)
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def deblur_dataset(input_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  sigma: float = 1.0,
                  amount: float = 1.0,
                  image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'),
                  preserve_structure: bool = True) -> dict:
    """
    Deblur all images in a dataset directory using unsharp masking.
    
    This function processes all images in the input directory and saves
    deblurred versions with the same filenames to the output directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save deblurred images
        sigma: Standard deviation for Gaussian filter (default: 1.0)
        amount: Enhancement amount coefficient (default: 1.0)
        image_extensions: Tuple of valid image file extensions
        preserve_structure: If True, preserves subdirectory structure
    
    Returns:
        Dictionary with statistics: {'processed': int, 'failed': int, 'total': int}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {'processed': 0, 'failed': 0, 'total': 0}
    
    # Find all image files
    image_files = []
    if preserve_structure:
        # Preserve directory structure
        for ext in image_extensions:
            image_files.extend(input_dir.rglob(f'*{ext}'))
            image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
    else:
        # Only process files in root directory
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    stats['total'] = len(image_files)
    
    print(f"Found {stats['total']} images to process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: sigma={sigma}, amount={amount}\n")
    
    # Process each image
    for img_path in image_files:
        # Calculate relative path from input_dir
        if preserve_structure:
            relative_path = img_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            # Create subdirectories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = output_dir / img_path.name
        
        # Process image
        if deblur_image_from_path(img_path, output_path, sigma=sigma, amount=amount):
            stats['processed'] += 1
            print(f"✓ Processed: {img_path.name}")
        else:
            stats['failed'] += 1
            print(f"✗ Failed: {img_path.name}")
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total images: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"{'='*50}")
    
    return stats


if __name__ == "__main__":
    input_directory = Path('images')
    output_directory = Path('deblur')
    
    deblur_dataset(
        input_dir=input_directory,
        output_dir=output_directory,
        sigma=2.0,      # Increased for better blur detection
        amount=1.5,     # Increased for stronger sharpening
        preserve_structure=True
    )

