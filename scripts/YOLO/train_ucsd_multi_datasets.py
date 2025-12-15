"""Train YOLO models on self-built UCSD dataset variants (train -> eval splits)."""

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
        help="Model to fine-tune.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Optional path to weights. Defaults to the path for --model.",
    )
    parser.add_argument("--device", default="1", help="Device string, e.g., '0', '1', '0,1'.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training/inference size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs"), help="Dir to store outputs.")
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "self_build_datasets"),
        help="Root containing self-built datasets.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Optional subset of dataset names to train, e.g., images_blur images_foggy.",
    )
    parser.add_argument("--save-preds", action="store_true", help="Save predicted images after training.")
    return parser.parse_args()


def discover_experiments(dataset_root: Path) -> list[tuple[str, Path, Path, Path]]:
    """Discover datasets under self_build_datasets with train/eval splits in images_* dirs."""
    experiments: list[tuple[str, Path, Path, Path]] = []
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
            train_dir = variant_dir / "train"
            eval_dir = variant_dir / "eval"
            if not train_dir.exists():
                print(f"Skipping {variant_dir}: missing train split at {train_dir}")
                continue
            if not eval_dir.exists():
                print(f"Skipping {variant_dir}: missing eval split at {eval_dir}")
                continue

            name = variant_dir.name
            if name in seen:
                raise RuntimeError(f"Duplicate experiment name '{name}' found at {variant_dir}.")
            seen.add(name)
            experiments.append((name, train_dir, eval_dir, labels_dir))

    return experiments


def _warn_if_mismatch(name: str, label_stems: set[str], image_stems: Iterable[str], split: str) -> None:
    image_set = set(image_stems)
    missing_imgs = label_stems - image_set
    missing_lbls = image_set - label_stems
    if missing_imgs or missing_lbls:
        print(
            f"Warning: {name} {split} mismatch. Missing images for {len(missing_imgs)} labels, "
            f"missing labels for {len(missing_lbls)} images."
        )


def prepare_training_experiment(
    name: str, train_dir: Path, eval_dir: Path, labels_dir: Path, auto_root: Path
) -> tuple[Path, Path, Path] | None:
    """Prepare list and YAML for a single dataset variant."""
    train_map = collect_images(train_dir)
    val_map = collect_images(eval_dir)
    if not train_map:
        print(f"Skipping {name}: no train images found in {train_dir}")
        return None
    if not val_map:
        print(f"Skipping {name}: no eval images found in {eval_dir}")
        return None

    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    train_common = sorted(set(train_map) & label_stems)
    val_common = sorted(set(val_map) & label_stems)
    if not train_common:
        print(f"Skipping {name}: no overlapping images and labels between {train_dir} and {labels_dir}")
        return None
    if not val_common:
        print(f"Skipping {name}: no overlapping images and labels between {eval_dir} and {labels_dir}")
        return None

    _warn_if_mismatch(name, label_stems, train_map.keys(), "train")
    _warn_if_mismatch(name, label_stems, val_map.keys(), "eval")

    train_images_link = auto_root / name / "train" / "images"
    train_labels_link = auto_root / name / "train" / "labels"
    val_images_link = auto_root / name / "val" / "images"
    val_labels_link = auto_root / name / "val" / "labels"
    for link, target in (
        (train_images_link, train_dir),
        (train_labels_link, labels_dir),
        (val_images_link, eval_dir),
        (val_labels_link, labels_dir),
    ):
        link.parent.mkdir(parents=True, exist_ok=True)
        ensure_symlink(link, target)

    train_list = auto_root / f"_auto_{name}_train.txt"
    val_list = auto_root / f"_auto_{name}_val.txt"
    train_list.write_text("\n".join(str(train_images_link / f"{stem}{train_map[stem].suffix}") for stem in train_common))
    val_list.write_text("\n".join(str(val_images_link / f"{stem}{val_map[stem].suffix}") for stem in val_common))

    data_cfg = {
        "path": str(labels_dir.parent),
        "train": str(train_list),
        "val": str(val_list),
        "test": str(val_list),
        "names": UCSD_NAMES,
    }
    data_path = auto_root / f"_auto_{name}_train.yaml"
    write_data_yaml(data_cfg, data_path)

    return train_list, val_list, data_path


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
        available = {name for name, _, _, _ in experiments}
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

    for name, train_dir, eval_dir, labels_dir in experiments:
        prepared = prepare_training_experiment(name, train_dir, eval_dir, labels_dir, auto_root)
        if not prepared:
            continue
        train_list, val_list, data_path = prepared

        run_name = f"{model_name}_{name}_train"
        print(f"\n=== Training {name} (train: {train_dir}, eval: {eval_dir}) ===")
        model = YOLO(str(weights_path))
        train_results = model.train(
            data=str(data_path),
            imgsz=args.imgsz,
            device=args.device,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            project=args.project,
            name=run_name,
            exist_ok=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            shear=1.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
        )
        print(f"{name} training finished. Results: {train_results}")

        best_weights = Path(args.project) / run_name / "weights" / "best.pt"
        if best_weights.exists():
            val_model = YOLO(str(best_weights))
            metrics = val_model.val(
                data=str(data_path),
                imgsz=args.imgsz,
                device=args.device,
                project=args.project,
                name=f"{run_name}_val",
                exist_ok=True,
            )
            result_dict = getattr(metrics, "results_dict", None) or {}
            print(f"{name} validation metrics (best weights): {result_dict}\n")

            if args.save_preds:
                pred_name = f"{run_name}_pred"
                print(f"Saving predictions for {name} from {val_list} ...")
                val_model.predict(
                    source=str(val_list),
                    imgsz=args.imgsz,
                    device=args.device,
                    project=args.project,
                    name=pred_name,
                    save=True,
                    exist_ok=True,
                )
                print(f"Predicted images saved to {Path(args.project) / pred_name}\n")
        else:
            print(f"Best weights not found for {name} at {best_weights}; skipping post-train val.")


if __name__ == "__main__":
    main()
