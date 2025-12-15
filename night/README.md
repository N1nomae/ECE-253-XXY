# Night Image Enhancement

This folder contains CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement for night images.

## Script

### `clahe.py`
Adaptive CLAHE enhancement that automatically adjusts parameters based on image brightness to improve visibility in low-light conditions.

**Usage:**
1. Update the configuration at the bottom of the file:
   - `input_folder`: Input image directory
   - `output_folder`: Output directory for enhanced images
   - `clip_limit`: Base clip limit for CLAHE (default: 4.0)
   - `tile_grid_size`: Grid size for CLAHE tiles (default: (8, 8))
2. Run: `python clahe.py`

## Adaptive Parameters

The script automatically adjusts CLAHE parameters based on image brightness:
- **Very dark images** (brightness < 50): `clip_limit * 1.5`, `tile=(6,6)`
- **Dark images** (brightness < 100): `clip_limit * 1.2`, `tile=(8,8)`
- **Normal/bright images**: Uses base parameters

## Requirements

- OpenCV (`cv2`)
- NumPy
- Pathlib (built-in)

Install with:
```bash
pip install opencv-python numpy
```

## Notes

- Processes all images in the input directory
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Enhancement is applied to the L channel in LAB color space to preserve color information

