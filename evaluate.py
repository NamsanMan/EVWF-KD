import torch
import numpy as np
import inspect
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F   # ★ 추가

from config import DATA


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _supports_depth_forward(model) -> bool:
    m = _unwrap_model(model)
    try:
        sig = inspect.signature(m.forward)
        return "depth" in sig.parameters
    except Exception:
        return False


def _forward_eval_model(model, imgs, eval_view: str = "lr", eval_depth_mode: str = "input"):
    """
    RGB-only / depth-fusion 모델 모두 지원.
    eval_view:
        "lr" -> (x_lr, x_hr, depth)에서 x_lr 사용
        "hr" -> (x_lr, x_hr, depth)에서 x_hr 사용
    """
    use_depth = _supports_depth_forward(model)

    if not isinstance(imgs, (tuple, list)):
        return model(imgs)

    if len(imgs) >= 3:
        x_lr, x_hr, depth = imgs[:3]
        x = x_lr if eval_view.lower() == "lr" else x_hr
        if use_depth:
            if eval_depth_mode.lower() == "zero":
                depth = torch.zeros_like(depth)
            return model(x, depth=depth)
        return model(x)

    if len(imgs) == 2 and use_depth:
        x, depth = imgs
        if eval_depth_mode.lower() == "zero":
            depth = torch.zeros_like(depth)
        return model(x, depth=depth)

    if len(imgs) == 2:
        x_lr, x_hr = imgs[:2]
        x = x_lr if eval_view.lower() == "lr" else x_hr
        return model(x)

    if len(imgs) == 1:
        return model(imgs[0])

    raise RuntimeError("Empty imgs tuple/list received in evaluation.")

@torch.inference_mode()
def evaluate_all(model, loader, device, eval_view: str = "lr", eval_depth_mode: str = "input"):
    model.eval()

    all_preds = []
    all_masks = []

    for imgs, masks in loader:
        imgs = _move_to_device(imgs, device)
        masks = masks.to(device, non_blocking=True)

        logits = _forward_eval_model(
            model,
            imgs,
            eval_view=eval_view,
            eval_depth_mode=eval_depth_mode,
        )  # (B,C,h,w)

        # ★ 핵심: GT mask 해상도에 맞춰 logits 업샘플 후 argmax
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        preds = torch.argmax(logits, dim=1)  # (B,H,W)

        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    # 1) 합치기
    preds_np = np.concatenate([p.flatten() for p in all_preds]).astype(np.int64)
    masks_np = np.concatenate([m.flatten() for m in all_masks]).astype(np.int64)

    # 2) 라벨 방어적 정규화
    oob_true = (masks_np != DATA.IGNORE_INDEX) & (
        (masks_np < 0) | (masks_np >= DATA.NUM_CLASSES)
    )
    masks_np[oob_true] = DATA.IGNORE_INDEX

    # 3) Void 제외
    valid = masks_np != DATA.IGNORE_INDEX
    masks_np = masks_np[valid]
    preds_np = preds_np[valid]

    # 4) 예측 클립
    if preds_np.size > 0:
        np.clip(preds_np, 0, DATA.NUM_CLASSES - 1, out=preds_np)

    # 5) Pixel Acc
    den = len(masks_np)
    pa = (np.sum(preds_np == masks_np) / den) if den > 0 else 0.0

    # 6) Confusion Matrix
    cm = confusion_matrix(masks_np, preds_np, labels=list(range(DATA.NUM_CLASSES)))

    # 7) IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = np.zeros(DATA.NUM_CLASSES, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=(union > 0))

    valid_classes_iou = [iou[c] for c in range(DATA.NUM_CLASSES) if c != DATA.IGNORE_INDEX and union[c] > 0]
    miou = np.nanmean(valid_classes_iou)

    per_class_iou = np.full(DATA.NUM_CLASSES, np.nan)
    for c in range(DATA.NUM_CLASSES):
        if c != DATA.IGNORE_INDEX and union[c] > 0:
            per_class_iou[c] = iou[c]

    return {
        "mIoU": miou,
        "PixelAcc": pa,
        "per_class_iou": per_class_iou,
        "confusion_matrix": cm
    }
