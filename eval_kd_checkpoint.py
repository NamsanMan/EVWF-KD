import argparse
import csv
import inspect
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained segmentation/KD student checkpoint without "
            "importing train_kd.py or constructing the KD engine."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, uses <result_root>/<project-name>/<ckpt-name>.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Experiment folder under the result root when --ckpt is omitted.",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="best_model.pth",
        help="Checkpoint filename used with --project-name. Default: best_model.pth.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Student model name. Default: config.KD.STUDENT_NAME.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Split root containing images/ and labels/. Ignored when --image-dir is given.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Evaluation image directory.",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default=None,
        help="Evaluation label directory. Defaults to config split label dir.",
    )
    parser.add_argument(
        "--depth-dir",
        type=str,
        default=None,
        help="Optional depth directory. Student evaluation is RGB-only by default; depth is used only when this is explicitly given.",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Keep RGB-only evaluation. This is the default and is kept for explicit command readability.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Split name used for default paths and output naming.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: <checkpoint_parent>/eval_<split>_<timestamp>.",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=5,
        help="Number of random prediction visualizations to save.",
    )
    parser.add_argument(
        "--vis-indices",
        type=str,
        default=None,
        help="Comma-separated dataset indices for visualization. Overrides random sampling.",
    )
    parser.add_argument(
        "--save-all-preds",
        action="store_true",
        help="Save color prediction PNGs for every sample.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size. Default: 1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default: 0.",
    )
    parser.add_argument(
        "--eval-view",
        type=str,
        default="lr",
        choices=("lr", "hr"),
        help="Input view used by evaluate.evaluate_all for tuple samples.",
    )
    parser.add_argument(
        "--eval-depth-mode",
        type=str,
        default="input",
        choices=("input", "zero"),
        help="Depth handling for depth-aware models.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=getattr(config.GENERAL, "SEED", 42),
        help="Random seed for visualization sampling.",
    )
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.ckpt:
        return Path(args.ckpt)

    base_dir = Path(config.GENERAL.BASE_DIR)
    if args.project_name:
        base_dir = base_dir.parent / args.project_name
    return base_dir / args.ckpt_name


def resolve_default_split_path(split: str, kind: str) -> Optional[Path]:
    attr = f"{split.upper()}_{kind.upper()}_DIR"
    value = getattr(config.DATA, attr, None)
    return Path(value) if value is not None else None


def resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path, Optional[Path]]:
    if args.data_dir:
        root = Path(args.data_dir)
        image_dir = root / "images"
        label_dir = root / "labels"
    else:
        image_dir = resolve_default_split_path(args.split, "img")
        label_dir = resolve_default_split_path(args.split, "label")

    if args.image_dir:
        image_dir = Path(args.image_dir)
    if args.label_dir:
        label_dir = Path(args.label_dir)

    if image_dir is None or label_dir is None:
        raise ValueError("Could not resolve image/label directories.")

    if args.depth_dir and not args.no_depth:
        depth_dir = Path(args.depth_dir)
    else:
        depth_dir = None

    return image_dir, label_dir, depth_dir


def apply_config_path_overrides(
    split: str,
    image_dir: Path,
    label_dir: Path,
    depth_dir: Optional[Path],
) -> None:
    prefix = split.upper()
    setattr(config.DATA, f"{prefix}_IMG_DIR", image_dir)
    setattr(config.DATA, f"{prefix}_LABEL_DIR", label_dir)
    setattr(config.DATA, f"{prefix}_DEPTH_DIR", depth_dir)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    ret = model.load_state_dict(state, strict=True)

    epoch = ckpt.get("epoch", "N/A") if isinstance(ckpt, dict) else "N/A"
    return {
        "epoch": epoch,
        "load_result": str(ret),
        "raw_checkpoint": ckpt,
    }


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def supports_depth_forward(model: torch.nn.Module) -> bool:
    try:
        sig = inspect.signature(unwrap_model(model).forward)
        return "depth" in sig.parameters
    except Exception:
        return False


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def add_batch_dim(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.unsqueeze(0)
    if isinstance(obj, (list, tuple)):
        return type(obj)(add_batch_dim(x) for x in obj)
    if isinstance(obj, dict):
        return {k: add_batch_dim(v) for k, v in obj.items()}
    return obj


def forward_eval_sample(
    model: torch.nn.Module,
    sample: Any,
    device: torch.device,
    eval_view: str,
    eval_depth_mode: str,
) -> torch.Tensor:
    from evaluate import _forward_eval_model

    batched = add_batch_dim(sample)
    batched = move_to_device(batched, device)
    logits = _forward_eval_model(
        model,
        batched,
        eval_view=eval_view,
        eval_depth_mode=eval_depth_mode,
    )
    return logits


def extract_display_image(sample: Any, eval_view: str = "lr") -> torch.Tensor:
    if torch.is_tensor(sample):
        return sample
    if not isinstance(sample, (tuple, list)):
        raise TypeError(f"Unsupported sample type: {type(sample)}")
    if len(sample) >= 3:
        return sample[0] if eval_view.lower() == "lr" else sample[1]
    if len(sample) == 2:
        a, b = sample
        if torch.is_tensor(b) and b.dim() == 3 and b.size(0) == 1:
            return a
        return a if eval_view.lower() == "lr" else b
    if len(sample) == 1:
        return sample[0]
    raise RuntimeError("Empty sample.")


def tensor_to_rgb_image(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().float().clone()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)
    img = (img * std + mean).clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def decode_segmap(label_mask: np.ndarray) -> np.ndarray:
    colors = np.asarray(config.DATA.CLASS_COLORS, dtype=np.uint8)
    ignore_index = int(config.DATA.IGNORE_INDEX)
    safe = np.asarray(label_mask).astype(np.int64, copy=True)
    invalid = (safe < 0) | (safe >= len(colors))
    safe[invalid] = ignore_index
    return colors[safe]


def overlay_prediction(image_rgb: np.ndarray, pred_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image_rgb.astype(np.float32)
    pred = pred_rgb.astype(np.float32)
    return ((1.0 - alpha) * image + alpha * pred).clip(0, 255).astype(np.uint8)


def save_image(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(config.DATA.NUM_CLASSES)
    plt.xticks(tick_marks, config.DATA.CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, config.DATA.CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=180)
    plt.close()


def save_visualization_grid(
    image_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    items = [
        ("Input", image_rgb),
        ("Ground Truth", gt_rgb),
        ("Prediction", pred_rgb),
        ("Overlay", overlay_rgb),
    ]
    for ax, (name, arr) in zip(axes, items):
        ax.imshow(arr)
        ax.set_title(name)
        ax.axis("off")
        ax.grid(False)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=180)
    plt.close(fig)


def nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.floating) and np.isnan(float(value)):
        return None
    return value


def write_metrics(
    out_dir: Path,
    ckpt_path: Path,
    model_name: str,
    image_dir: Path,
    label_dir: Path,
    depth_dir: Optional[Path],
    epoch: Any,
    metrics: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_iou = np.asarray(metrics["per_class_iou"], dtype=np.float64)
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)

    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("KD checkpoint evaluation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Checkpoint : {ckpt_path}\n")
        f.write(f"Model      : {model_name}\n")
        f.write(f"Epoch      : {epoch}\n")
        f.write(f"Images     : {image_dir}\n")
        f.write(f"Labels     : {label_dir}\n")
        f.write(f"Depths     : {depth_dir if depth_dir is not None else 'None'}\n\n")
        f.write(f"Test mIoU      : {float(metrics['mIoU']):.6f}\n")
        f.write(f"Test Pixel Acc : {float(metrics['PixelAcc']):.6f}\n\n")
        f.write("Per-class IoU\n")
        for name, iou in zip(config.DATA.CLASS_NAMES, per_class_iou):
            text = "nan" if np.isnan(iou) else f"{iou:.6f}"
            f.write(f"  {name}: {text}\n")

    json_payload = {
        "checkpoint": str(ckpt_path),
        "model_name": model_name,
        "epoch": epoch,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "depth_dir": str(depth_dir) if depth_dir is not None else None,
        "mIoU": float(metrics["mIoU"]),
        "PixelAcc": float(metrics["PixelAcc"]),
        "per_class_iou": [nan_to_none(float(x)) for x in per_class_iou],
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    with (out_dir / "per_class_iou.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "IoU"])
        for name, iou in zip(config.DATA.CLASS_NAMES, per_class_iou):
            writer.writerow([name, "" if np.isnan(iou) else float(iou)])

    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_confusion_matrix_png(cm, out_dir / "confusion_matrix.png", "Test Confusion Matrix")


def parse_vis_indices(indices: Optional[str], dataset_len: int, num_vis: int, seed: int) -> list[int]:
    if indices:
        parsed = [int(x.strip()) for x in indices.split(",") if x.strip()]
        return [idx for idx in parsed if 0 <= idx < dataset_len]
    rng = random.Random(seed)
    k = min(max(num_vis, 0), dataset_len)
    return rng.sample(range(dataset_len), k) if k > 0 else []


@torch.inference_mode()
def save_prediction_visualizations(
    model: torch.nn.Module,
    dataset: Any,
    out_dir: Path,
    device: torch.device,
    eval_view: str,
    eval_depth_mode: str,
    indices: Iterable[int],
    save_all_preds: bool = False,
) -> None:
    vis_dir = out_dir / "visualizations"
    pred_dir = out_dir / "predictions_color"
    raw_pred_dir = out_dir / "predictions_label"
    vis_dir.mkdir(parents=True, exist_ok=True)

    all_indices = range(len(dataset)) if save_all_preds else indices

    for idx in all_indices:
        sample, mask_t = dataset[idx]
        logits = forward_eval_sample(model, sample, device, eval_view, eval_depth_mode)
        if logits.shape[-2:] != mask_t.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=mask_t.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pred_idx = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        filename = dataset.files[idx] if hasattr(dataset, "files") else f"{idx:04d}.png"
        stem = Path(filename).stem

        pred_rgb = decode_segmap(pred_idx)
        if save_all_preds:
            save_image(pred_dir / filename, pred_rgb)
            save_image(raw_pred_dir / filename, pred_idx)

        if idx not in set(indices):
            continue

        image_t = extract_display_image(sample, eval_view=eval_view)
        image_rgb = tensor_to_rgb_image(image_t)
        gt_idx = mask_t.detach().cpu().numpy().astype(np.int64)
        gt_rgb = decode_segmap(gt_idx)
        overlay_rgb = overlay_prediction(image_rgb, pred_rgb)

        sample_dir = vis_dir / stem
        save_image(sample_dir / "input.png", image_rgb)
        save_image(sample_dir / "ground_truth_color.png", gt_rgb)
        save_image(sample_dir / "prediction_color.png", pred_rgb)
        save_image(sample_dir / "prediction_overlay.png", overlay_rgb)
        save_image(sample_dir / "prediction_label.png", pred_idx)
        save_visualization_grid(
            image_rgb,
            gt_rgb,
            pred_rgb,
            overlay_rgb,
            sample_dir / "comparison.png",
            title=filename,
        )
        print(f"[Saved visualization] {sample_dir / 'comparison.png'}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = resolve_checkpoint(args)
    image_dir, label_dir, depth_dir = resolve_data_paths(args)
    apply_config_path_overrides(args.split, image_dir, label_dir, depth_dir)

    from data_loader import CamVidDataset, SegmentationTransform
    from evaluate import evaluate_all
    from models import create_model

    model_name = args.model_name or config.KD.STUDENT_NAME
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 기본 출력 위치는 결과 디렉터리다. checkpoint 옆에 쓰면 배포용 weights 폴더가
    # 평가 산출물로 지저분해진다.
    out_dir = (Path(args.out_dir) if args.out_dir
               else config.GENERAL.BASE_DIR / f"eval_{args.split}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    device = get_device(args.device)
    dataset = CamVidDataset(
        images_dir=str(image_dir),
        masks_dir=str(label_dir),
        file_list=getattr(config.DATA, "FILE_LIST", None),
        transform=SegmentationTransform(config.DATA.INPUT_RESOLUTION),
        teacher_images_dir=None,
        depth_dir=str(depth_dir) if depth_dir is not None else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_model(model_name).to(device)
    load_info = load_model_checkpoint(model, ckpt_path, device)
    model.eval()

    print(f"Checkpoint : {ckpt_path}")
    print(f"Model      : {model_name}")
    print(f"Epoch      : {load_info['epoch']}")
    print(f"Images     : {image_dir}")
    print(f"Labels     : {label_dir}")
    print(f"Depths     : {depth_dir if depth_dir is not None else 'None'}")
    print(f"Output     : {out_dir}")

    metrics = evaluate_all(
        model,
        loader,
        device,
        eval_view=args.eval_view,
        eval_depth_mode=args.eval_depth_mode,
    )
    print(
        f"Test mIoU: {float(metrics['mIoU']):.4f}, "
        f"Test Pixel Acc: {float(metrics['PixelAcc']):.4f}"
    )

    write_metrics(
        out_dir=out_dir,
        ckpt_path=ckpt_path,
        model_name=model_name,
        image_dir=image_dir,
        label_dir=label_dir,
        depth_dir=depth_dir,
        epoch=load_info["epoch"],
        metrics=metrics,
    )

    indices = parse_vis_indices(args.vis_indices, len(dataset), args.num_vis, args.seed)
    save_prediction_visualizations(
        model=model,
        dataset=dataset,
        out_dir=out_dir,
        device=device,
        eval_view=args.eval_view,
        eval_depth_mode=args.eval_depth_mode,
        indices=indices,
        save_all_preds=args.save_all_preds,
    )

    with (out_dir / "run_config.txt").open("w", encoding="utf-8") as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")

    print(f"Saved evaluation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
