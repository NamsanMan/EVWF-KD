"""
Train 부터 test 후 시각화 까지 end-to-end를 위한 main.py
"""

import os
import random
import inspect
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import config


def set_seed(seed):
    """
    재현성을 위해 시드를 고정하는 함수
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    # CuDNN 결정론적 연산 활성화
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"▶ Seed is fixed to {seed}")


set_seed(config.GENERAL.SEED)

import train_teacher
import data_loader
import evaluate


def decode_segmap(label_mask):
    """
    label_mask: 2D numpy array (H×W), 값은 [0..n_classes-1]
    return: 3D numpy array (H×W×3), dtype=uint8
    """
    return config.DATA.CLASS_COLORS[label_mask]


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _supports_depth_forward(model) -> bool:
    """
    모델 forward 시그니처에 depth 인자가 있는지 확인.
    RGB-only 모델 / depth-fusion 모델 모두 지원하기 위함.
    """
    m = _unwrap_model(model)
    try:
        sig = inspect.signature(m.forward)
        return "depth" in sig.parameters
    except Exception:
        return False


def _tensor_to_vis_image(img_t: torch.Tensor) -> np.ndarray:
    """
    정규화된 (3,H,W) 텐서를 시각화용 uint8 RGB 이미지로 복원.
    """
    img_vis = img_t.detach().cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_vis.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_vis.dtype).view(3, 1, 1)
    img_vis = (img_vis * std + mean).clamp(0.0, 1.0)
    img_np = (img_vis.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img_np


def _extract_eval_sample(sample, model, eval_view: str = "lr"):
    """
    data_loader.test_dataset[idx]가 반환하는 다양한 형식을 안전하게 해석.

    지원 형식:
      - image
      - (image, depth)
      - (image, teacher_image)
      - (image, teacher_image, depth)

    반환:
      img_t:   학생/LR 관점에서 평가할 입력 이미지 텐서
      depth_t: depth 텐서 또는 None
    """
    use_depth = _supports_depth_forward(model)

    if not isinstance(sample, (tuple, list)):
        return sample, None

    if len(sample) >= 3:
        # 일반적으로 (x_lr, x_hr, depth)
        img_t = sample[1] if eval_view.lower() == "hr" else sample[0]
        depth_t = sample[2]
        return img_t, depth_t

    if len(sample) == 2:
        a, b = sample

        # depth-aware 모델이면 두 번째 항이 depth일 가능성을 우선 고려
        # depth: (1,H,W), teacher RGB: (3,H,W)
        if use_depth and torch.is_tensor(b) and b.dim() == 3 and b.size(0) == 1:
            return a, b

        # RGB-only 경로 또는 (image, teacher_image) 같은 경우
        return a, None

    if len(sample) == 1:
        return sample[0], None

    raise RuntimeError("Unsupported empty sample encountered in visualization.")


def main():
    # 랜덤 이미지 시각화 할때는 seed 고정 영향 안받게 함
    visual_random = random.Random()

    if config.TRAIN.USE_CHECKPOINT:
        # train 안하고 checkpoint만 로드할 때
        checkpoint_name = "best_model.pth"
        best_ckpt = config.GENERAL.BASE_DIR / checkpoint_name
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")
    else:
        # 1) 학습 수행
        best_ckpt = train_teacher.run_training(num_epochs=config.TRAIN.EPOCHS)

    # 최종 best checkpoint 강제 사용
    checkpoint_name = "best_model.pth"
    best_ckpt = config.GENERAL.BASE_DIR / checkpoint_name
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")

    # 2) test를 위해 베스트 체크포인트 로드(student)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_teacher.model
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    epoch_num = ckpt["epoch"]
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # 3) 테스트셋 전체에 대해 mIoU / Pixel Accuracy 계산
    # evaluate.py가 depth-aware로 수정되어 있다는 전제
    eval_view = getattr(train_teacher, "EVAL_VIEW", "lr")
    eval_depth_mode = getattr(train_teacher, "EVAL_DEPTH_MODE", "input")
    metrics = evaluate.evaluate_all(
        model,
        data_loader.test_loader,
        device,
        eval_view=eval_view,
        eval_depth_mode=eval_depth_mode,
    )
    test_miou = metrics["mIoU"]
    test_pa = metrics["PixelAcc"]
    print(
        f"▶ Loaded model from epoch {epoch_num}, "
        f"Test mIoU: {test_miou:.4f}, Test Pixel Acc: {test_pa:.4f}"
    )

    per_cls_iou = metrics["per_class_iou"]
    df_test_iou = pd.DataFrame({
        "Class": config.DATA.CLASS_NAMES,
        "IoU": per_cls_iou
    })
    test_iou_path = config.GENERAL.LOG_DIR / f"test_iou_epoch_{epoch_num}.csv"
    df_test_iou.to_csv(test_iou_path, index=False)

    # 3.2) 테스트셋 confusion matrix 계산 및 저장
    cm_test = metrics["confusion_matrix"]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_test, interpolation="nearest")
    plt.title(f"Test Confusion Matrix (Epoch {epoch_num})")
    plt.colorbar()
    tick_marks = np.arange(config.DATA.NUM_CLASSES)
    plt.xticks(tick_marks, config.DATA.CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, config.DATA.CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    test_cm_path = config.GENERAL.LOG_DIR / f"test_confusion_matrix_epoch_{epoch_num}.png"
    plt.savefig(str(test_cm_path), bbox_inches="tight")
    plt.close()

    # 4) 결과 파일에 기록
    results_txt = config.GENERAL.BASE_DIR / "results.txt"
    os.makedirs(results_txt.parent, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(results_txt, "a", encoding="utf-8") as f:
        f.write(f"[{now}] Test mIoU: {test_miou:.4f}, Test PA: {test_pa:.4f}\n")

    # 5) 랜덤 5장 시각화
    output_dir = config.GENERAL.BASE_DIR / "images"
    os.makedirs(output_dir, exist_ok=True)

    dataset = data_loader.test_dataset
    k = min(5, len(dataset))
    sampled_indices = visual_random.sample(range(len(dataset)), k)

    for idx in sampled_indices:
        sample, mask_t = dataset[idx]
        img_t, depth_t = _extract_eval_sample(sample, model, eval_view=eval_view)

        # 예측 진행
        with torch.no_grad():
            x_in = img_t.unsqueeze(0).to(device)
            if _supports_depth_forward(model):
                if eval_depth_mode == "zero":
                    d_in = torch.zeros(1, 1, x_in.size(2), x_in.size(3),
                                       dtype=x_in.dtype, device=device)
                elif depth_t is not None:
                    d_in = depth_t.unsqueeze(0).to(device)
                else:
                    # depth 파일이 없을 때 zero depth fallback
                    d_in = torch.zeros(1, 1, x_in.size(2), x_in.size(3),
                                       dtype=x_in.dtype, device=device)
                pred_logits = model(x_in, depth=d_in)
            else:
                pred_logits = model(x_in)

            pred_idx = pred_logits.argmax(dim=1).squeeze().cpu().numpy()

        img_np = _tensor_to_vis_image(img_t)
        mask_idx = mask_t.numpy()

        # 디코딩
        mask_rgb = decode_segmap(mask_idx)
        pred_rgb = decode_segmap(pred_idx)

        H, W, _ = img_np.shape
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_np)
        axes[0].set_title(f"Original ({W}x{H})")
        axes[0].axis("off")
        axes[0].grid(False)

        axes[1].imshow(mask_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[1].grid(False)

        axes[2].imshow(pred_rgb)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        axes[2].grid(False)

        plt.tight_layout()

        if hasattr(dataset, "files") and idx < len(dataset.files):
            stem = Path(dataset.files[idx]).stem
        else:
            stem = f"{idx:04d}"

        save_path = output_dir / f"viz_{stem}.png"
        fig.savefig(str(save_path), bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
