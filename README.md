# ECE-253-XXY

ECE-253 final project repository for restoring and enhancing driving scenes captured under challenging conditions (blur, fog, and low light) and evaluating downstream detection with YOLO.

![Project workflow](images/work_flow.png)

## Getting Started
- Python 3.9+ recommended. GPU + CUDA speed up the PyTorch pieces.
- Install PyTorch appropriate for your hardware (see https://pytorch.org/). Example (CPU):
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- Common libraries used across scripts:
  ```bash
  pip install ultralytics opencv-python pillow numpy tqdm pyyaml requests
  ```
- Additional dehazing deps:
  ```bash
  pip install -r foggy/requirements.txt
  ```

## Repository Layout
- `blur/DeblurGAN/` – DeblurGAN inference for severe motion blur.
- `blur/unsharp_masking/` – Lightweight unsharp masking deblurring.
- `foggy/` – Dehazing with Dark Channel Prior (`dcp.py`) and FFANet (`ffanet.py`).
- `night/` – Low-light enhancement via adaptive CLAHE (`clahe.py`) and Zero-DCE (`enhance_zero_dce.py`).
- `scripts/YOLO/` – Train/evaluate YOLO models on self-built dataset variants.
- `scripts/dataset/` – Utilities for data prep/augmentation (fog synthesis, motion blur, resizing, label conversion, visualization).
- `images/work_flow.png` – High-level pipeline diagram.

## Image Enhancement Modules
### DeblurGAN (severe blur)
1. Place blurry inputs in `blur/DeblurGAN/my_blur_images/`.
2. Ensure the generator checkpoint exists at `blur/DeblurGAN/checkpoints/experiment_name/latest_net_G.pth`.
3. Run:
   ```bash
   cd blur/DeblurGAN
   python test.py --dataroot ./my_blur_images --model test --dataset_mode single --learn_residual \
     --name experiment_name --display_id 0 --resize_or_crop none
   ```
4. Deblurred results are written to `blur/DeblurGAN/results/experiment_name/deblurred/`.

### Unsharp Masking (mild blur)
1. Add inputs to `blur/unsharp_masking/images/`.
2. Run `python blur/unsharp_masking/deblur_images.py`.
3. Outputs land in `blur/unsharp_masking/deblur/` (default `sigma=2.0`, `amount=1.5`).

### Foggy Scene Dehazing
- `foggy/dcp.py` – Dark Channel Prior baseline. Configure `INPUT_DIR` and `OUTPUT_DIR` in the script, then run `python foggy/dcp.py`.
- `foggy/ffanet.py` – Feature Fusion Attention Network. Set `INPUT_DIR`, `OUTPUT_DIR`, and optionally `MODEL_PATH`, then run `python foggy/ffanet.py`. Falls back to enhanced DCP if no pretrained weights are present.
Both scripts process every `.jpg/.jpeg/.png` in the input directory.

### Night / Low-Light Enhancement
- `night/clahe.py` – Adaptive CLAHE with auto-tuned parameters. Set `input_folder` / `output_folder` at the bottom of the script and run `python night/clahe.py`.
- `night/enhance_zero_dce.py` – Zero-DCE inference; downloads official weights on first run. Example:
  ```bash
  python night/enhance_zero_dce.py --input /path/to/images --output /path/to/enhanced --device auto
  ```
  Uses CUDA when available; add `--overwrite` to replace existing outputs.

## YOLO Training and Evaluation
The scripts in `scripts/YOLO/` expect a self-built dataset organized as:
```
self_build_datasets/
  <dataset_group>/
    labels/*.txt               # YOLO format, class 0 = car
    images_<variant>/train/*.jpg|.png
    images_<variant>/eval/*.jpg|.png
```
Each `images_<variant>` directory (e.g., `images_blur`, `images_foggy`, `images_lowlight`) is treated as one experiment. Symlinks and split lists are generated automatically.

- Train across all discovered variants:
  ```bash
  cd scripts/YOLO
  python train_ucsd_multi_datasets.py --dataset-root /path/to/self_build_datasets \
    --model kitti-YOLOv11s --epochs 20 --batch 8 --imgsz 640 --device 0 --save-preds
  ```
  The script will fine-tune one model per variant, validate, and optionally save predictions.

- Evaluate existing weights on the eval split(s):
  ```bash
  python eval_ucsd_multi_datasets.py --dataset-root /path/to/self_build_datasets \
    --model kitti-YOLOv11s --imgsz 640 --device 0 --save-preds
  ```
  Use `--experiments images_blur images_foggy` to limit to specific variants. Metrics and predictions are saved under `scripts/YOLO/runs/`.

## Dataset Utility Scripts (`scripts/dataset/`)
- `add_fog.py` – Depth-aware fog synthesis with MiDaS; edit hardcoded paths before running.
- `motionblur.py` – Apply configurable linear motion blur to all images and copy labels.
- `convert.py` – Resize + center-crop images to 1224×370 (KITTI-friendly) while maintaining aspect ratio.
- `polygon2yolo.py` – Convert normalized polygon annotations to YOLO bboxes, filtering invalid/tiny boxes.
- `draw_boxes.py` – Visualize YOLO labels overlaid on images to verify annotations.
- `rename_files.py` – Sequentially rename images (and matching labels) to `001.*`, `002.*`, … to avoid collisions.

The link for dataset: ` https://drive.google.com/file/d/1OK1KIhkeaz5zaT50Xh7ZZMkESWZ2V9jw/view?usp=share_link`


## Notes
- Most scripts assume local paths defined at the top of the file—update them to your dataset locations before running.
- For GPU inference/training, confirm CUDA availability with `torch.cuda.is_available()`.
