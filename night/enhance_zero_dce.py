#!/usr/bin/env python3
"""
Enhance low-light images with the Zero-DCE model.

The script downloads the official pre-trained weights from
https://github.com/Li-Chongyi/Zero-DCE and runs inference on a folder
of images. Outputs keep the same relative paths under the output
directory.
"""

import argparse
from pathlib import Path
import sys
from typing import Iterable, List

import numpy as np
from PIL import Image
import requests

try:  # Optional progress bar; falls back to a plain iterator.
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - dependency hint for first run
    raise SystemExit(
        "PyTorch is required. Install CPU wheels with "
        "`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu` "
        "and re-run this script."
    ) from exc


MODEL_URL = "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth"
DEFAULT_INPUT = Path("/scratch/tpang/yuanzhe/Kitti/self_build_datasets/ucsd_night/images_lowlight")
DEFAULT_OUTPUT = Path("/scratch/tpang/yuanzhe/Kitti/self_build_datasets/ucsd_night/images_zero_dce")
CACHE_DIR = Path(".zero_dce_cache")


class EnhanceNetNoPool(nn.Module):
    """Zero-DCE network (copied from the official repo)."""

    def __init__(self) -> None:
        super().__init__()
        number_f = 32
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r


def download_weights(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target

    print(f"Downloading Zero-DCE weights to {target} ...")
    with requests.get(MODEL_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(target, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
    return target


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device, weights_path: Path) -> EnhanceNetNoPool:
    weights = download_weights(weights_path)
    model = EnhanceNetNoPool().to(device)
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts and p.is_file())


def to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_tensor_image(tensor: torch.Tensor, output_path: Path) -> None:
    array = tensor.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array * 255).round().astype("uint8")).save(output_path)


def enhance_folder(input_dir: Path, output_dir: Path, device: torch.device, overwrite: bool = False) -> None:
    images = list_images(input_dir)
    if not images:
        print(f"No images found under {input_dir}")
        return

    model = load_model(device, CACHE_DIR / "Epoch99.pth")
    with torch.no_grad():
        for image_path in tqdm(images, desc="Enhancing", unit="img"):
            relative = image_path.relative_to(input_dir)
            out_path = output_dir / relative
            if out_path.exists() and not overwrite:
                continue
            img = Image.open(image_path)
            _, enhanced, _ = model(to_tensor(img, device))
            save_tensor_image(enhanced, out_path)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Zero-DCE on a folder of images.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Folder with low-light images.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where enhanced images will be written.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Select inference device. Default picks CUDA when available.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run and replace existing outputs.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    enhance_folder(args.input, args.output, device, overwrite=args.overwrite)


if __name__ == "__main__":
    main(sys.argv[1:])
