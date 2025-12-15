"""Evaluate YOLO models on self-built UCSD dataset variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

REPO_ROOT = Path(__file__).resolve().parent
UCSD_NAMES: Dict[int, str] = {0: "car"}
MODEL_CHOICES = {
    "kitti-YOLOv11n": REPO_ROOT / "runs" / "kitti-YOLOv11n" / "weights" / "best.pt",
    "kitti-YOLOv11s": REPO_ROOT / "runs" / "kitti-YOLOv11s" / "weights" / "best.pt",
}
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def write_data_yaml(cfg: dict, path: Path) -> None:
    """Write a temporary data YAML for Ultralytics."""
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def collect_images(root: Path) -> dict[str, Path]:
    """Collect images in a directory keyed by stem, supporting common extensions."""
    images = {}
    for p in root.iterdir():
        if p.suffix.lower() in VALID_IMAGE_EXTS and p.is_file():
            images[p.stem] = p
    return images


def ensure_symlink(link: Path, target: Path) -> None:
    """Create or replace a symlink."""
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target, target_is_directory=target.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES),
        default="kitti-YOLOv11s",
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Optional path to weights. Defaults to the path for --model.",
    )
    parser.add_argument("--device", default="1", help="Device string, e.g., '0', '1', '0,1'.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size.")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs"), help="Dir to store outputs.")
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "self_build_datasets"),
        help="Root containing self-built datasets.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Optional subset of dataset names to evaluate, e.g., images_blur images_foggy.",
    )
    parser.add_argument("--save-preds", action="store_true", help="Save predicted images with boxes.")
    return parser.parse_args()


def discover_experiments(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    """Discover datasets under self_build_datasets (one eval split per images_* dir)."""
    experiments: list[tuple[str, Path, Path]] = []
    seen: set[str] = set()

    for group_dir in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if not group_dir.is_dir():
            continue
        labels_dir = group_dir / "labels"
        if not labels_dir.exists():
            print(f"Skipping {group_dir.name}: missing labels dir {labels_dir}")
            continue

        variant_dirs = sorted(
            (p for p in group_dir.iterdir() if p.is_dir() and p.name.startswith("images")),
            key=lambda p: p.name,
        )
        for variant_dir in variant_dirs:
            eval_dir = variant_dir / "eval"
            if not eval_dir.exists():
                print(f"Skipping {variant_dir}: missing eval split at {eval_dir}")
                continue

            name = variant_dir.name
            if name in seen:
                raise RuntimeError(f"Duplicate experiment name '{name}' found at {variant_dir}.")
            seen.add(name)
            experiments.append((name, eval_dir, labels_dir))

    return experiments


def _warn_if_mismatch(name: str, label_stems: set[str], image_stems: Iterable[str]) -> None:
    image_set = set(image_stems)
    missing_imgs = label_stems - image_set
    missing_lbls = image_set - label_stems
    if missing_imgs or missing_lbls:
        print(
            f"Warning: {name} mismatch. Missing images for {len(missing_imgs)} labels, "
            f"missing labels for {len(missing_lbls)} images."
        )


def prepare_experiment(name: str, eval_dir: Path, labels_dir: Path, auto_root: Path) -> tuple[Path, Path] | None:
    """Prepare list and YAML for a single dataset variant."""
    image_map = collect_images(eval_dir)
    if not image_map:
        print(f"Skipping {name}: no images found in {eval_dir}")
        return None

    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    common_names = sorted(set(image_map) & label_stems)
    if not common_names:
        print(f"Skipping {name}: no overlapping images and labels between {eval_dir} and {labels_dir}")
        return None

    _warn_if_mismatch(name, label_stems, image_map.keys())

    images_link = auto_root / name / "images"
    labels_link = auto_root / name / "labels"
    images_link.parent.mkdir(parents=True, exist_ok=True)
    labels_link.parent.mkdir(parents=True, exist_ok=True)
    ensure_symlink(images_link, eval_dir)
    ensure_symlink(labels_link, labels_dir)

    list_path = auto_root / f"_auto_{name}_val.txt"
    list_path.write_text("\n".join(str(images_link / f"{stem}{image_map[stem].suffix}") for stem in common_names))

    data_cfg = {
        "path": str(labels_dir.parent),
        "train": str(list_path),
        "val": str(list_path),
        "test": str(list_path),
        "names": UCSD_NAMES,
    }
    data_path = auto_root / f"_auto_{name}.yaml"
    write_data_yaml(data_cfg, data_path)

    return list_path, data_path


def main() -> None:
    args = parse_args()

    SETTINGS.update(runs_dir=args.project)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    auto_root = REPO_ROOT / "_auto_eval"
    auto_root.mkdir(exist_ok=True)

    experiments = discover_experiments(dataset_root)
    if args.experiments:
        requested = set(args.experiments)
        available = {name for name, _, _ in experiments}
        missing = requested - available
        if missing:
            raise ValueError(f"Requested experiments not found: {sorted(missing)}")
        experiments = [exp for exp in experiments if exp[0] in requested]

    if not experiments:
        raise RuntimeError("No valid experiments discovered under self_build_datasets.")

    weights_path = Path(args.weights) if args.weights else MODEL_CHOICES[args.model]
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    model_name = args.model
    model = YOLO(str(weights_path))

    for name, eval_dir, labels_dir in experiments:
        prepared = prepare_experiment(name, eval_dir, labels_dir, auto_root)
        if not prepared:
            continue
        list_path, data_path = prepared

        run_name = f"{model_name}_{name}"
        print(f"\n=== Evaluating {name} ({eval_dir}) ===")
        metrics = model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=run_name,
            exist_ok=True,
        )
        result_dict = getattr(metrics, "results_dict", None) or {}
        print(f"{name} metrics: {result_dict}\n")

        if args.save_preds:
            pred_name = f"{run_name}-pred"
            print(f"Saving predictions for {name} from {list_path} ...")
            model.predict(
                source=str(list_path),
                imgsz=args.imgsz,
                device=args.device,
                project=args.project,
                name=pred_name,
                save=True,
                exist_ok=True,
            )
            print(f"Predicted images saved to {Path(args.project) / pred_name}\n")


if __name__ == "__main__":
    main()
