# Image Dehazing

This folder contains two image dehazing methods for processing foggy images.

## Scripts

### `dcp.py`
Dark Channel Prior (DCP) based dehazing algorithm. 

**Usage:**
1. Update the configuration at the top of the file:
   - `INPUT_DIR`: Input image directory
   - `OUTPUT_DIR`: Output directory for dehazed images
2. Run: `python dcp.py`

### `ffanet.py`
Feature Fusion Attention Network (FFANet) for image dehazing.

**Usage:**
1. Update the configuration at the top of the file:
   - `INPUT_DIR`: Input image directory
   - `OUTPUT_DIR`: Output directory for dehazed images
   - `MODEL_PATH`: Path to pretrained model (optional)
2. Run: `python ffanet.py`

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Notes

- Both scripts process all images in the input directory
- Supported formats: `.jpg`, `.jpeg`, `.png`
- If no pretrained model is available, `ffanet.py` automatically uses the enhanced DCP method

