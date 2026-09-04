import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import shutil
import math
import numpy as np
import inspect
from tqdm import tqdm
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # ← AMP

import data_loader
import config
from models import create_model
import evaluate

from kd_engines import create_kd_engine

# 보기 싫은 로그 숨김
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

LOSS_KEY_DISPLAY_OVERRIDES = {
    "total": "Total Loss",
    "ce_hr": "CE HR Loss",
    "ce_lr": "CE LR Loss",
    "lapc": "LCFC Loss",
    "lapc_raw": "LCFC Raw",
}

# CSV 로그에 남길 항목. 엔진은 이 밖에도 다수의 진단용 scalar 를 반환하지만
# 학습에는 쓰이지 않으므로 기록하지 않는다.
CORE_LOSS_KEYS = list(LOSS_KEY_DISPLAY_OVERRIDES.keys())


SCALAR_LOSS_KEYS: List[str] = []
LOSS_KEY_TO_HEADER: Dict[str, str] = {}
LOSS_HEADER_ORDER: List[str] = []


def _is_scalar_loss_value(value) -> bool:
    if isinstance(value, torch.Tensor):
        return value.dim() == 0
    return isinstance(value, (int, float))


def _loss_value_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _display_name_for_loss(key: str) -> str:
    if key in LOSS_KEY_DISPLAY_OVERRIDES:
        return LOSS_KEY_DISPLAY_OVERRIDES[key]
    pretty = key.replace("_", " ").title()
    if "loss" not in key.lower():
        pretty = f"{pretty} Loss"
    return pretty


# ── model 설정 ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
# teacher 와 student 가 같은 모델이면 인스턴스를 공유한다.
# ────────────────────────────────────────────────────────────────
student = create_model(config.KD.STUDENT_NAME).to(device)

if (config.KD.TEACHER_NAME == config.KD.STUDENT_NAME) and (not config.KD.FREEZE_TEACHER):
    # 인스턴스 공유 → 메모리 절약 + weight 공유 보장
    teacher = student
    print("Shared-instance mode: teacher and student share the same model instance.")
else:
    teacher = create_model(config.KD.TEACHER_NAME).to(device)

model = student

if config.KD.FREEZE_TEACHER:
    # checkpoint 없이 freeze 하면 ImageNet 가중치로 distill 하게 되어
    # 그럴듯하지만 틀린 student 가 나온다. 경고 대신 즉시 중단한다.
    if not Path(config.TEACHER_CKPT).exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {config.TEACHER_CKPT}\n"
            "SWEEP_TEACHER_CKPT 를 teacher 학습 결과의 best_model.pth 로 지정하십시오."
        )
    try:
        ckpt_path = Path(config.TEACHER_CKPT)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            # --- STRICT teacher checkpoint validation ---
            sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

            # DataParallel 호환 (module. prefix 제거)
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

            ret = teacher.load_state_dict(sd, strict=True)
            print(
                f"▶ Teacher checkpoint loaded OK (strict=True): {ckpt_path} "
                f"| incompatible_keys={ret}"
            )
        else:
            print(f"⚠️ WARNING: Teacher checkpoint not found at {ckpt_path}. Using ImageNet pretrained weights.")
    except Exception as e:
        raise RuntimeError(f"❌ Teacher checkpoint load FAILED (strict=True): {ckpt_path}\n{e}")

    # Freeze the teacher so only the student updates during KD.
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()


# ── KD 엔진 구성 ───────────────────────────────────
kd_engine = create_kd_engine(config.KD, teacher, student).to(device)


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
    """
    RGB-only wrapper와 depth-fusion wrapper를 모두 지원하기 위한 helper.
    forward 시그니처에 depth 인자가 있으면 True.
    """
    m = _unwrap_model(model)
    try:
        sig = inspect.signature(m.forward)
        return "depth" in sig.parameters
    except Exception:
        return False


def _forward_eval_model(model, imgs, eval_view: str = "lr", eval_depth_mode: str = "input"):
    """
    기존 RGB-only 모델과 depth-fusion 모델을 모두 지원.

    imgs 형태:
      - Tensor
      - (x_lr, x_hr)
      - (x_lr, x_hr, depth)
      - 더 긴 tuple/list여도 앞 3개만 사용
    """
    use_depth = _supports_depth_forward(model)

    # RGB-only 단일 입력
    if not isinstance(imgs, (tuple, list)):
        return model(imgs)

    # tuple/list 입력
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
        return model(x, depth=depth)

    if len(imgs) == 2:
        x_lr, x_hr = imgs[:2]
        x = x_lr if eval_view.lower() == "lr" else x_hr
        return model(x)

    if len(imgs) == 1:
        return model(imgs[0])

    raise RuntimeError("Empty imgs tuple/list received in evaluation.")

# --- build KD projections (dry-run) so their params are included in optimizer ---
imgs0, masks0 = next(iter(data_loader.train_loader))
imgs0 = _move_to_device(imgs0, device)
masks0 = masks0.to(device, non_blocking=True)
with torch.no_grad():
    dry_run_out = kd_engine.compute_losses(imgs0, masks0, device)

if "total" not in dry_run_out:
    raise KeyError("KD engine must return a 'total' loss entry.")

if hasattr(kd_engine, "get_primary_model") and callable(kd_engine.get_primary_model):
    model = kd_engine.get_primary_model()
else:
    model = student
EVAL_VIEW = getattr(
    kd_engine,
    "primary_eval_view",
    "hr" if getattr(kd_engine, "trains_teacher_only", False) else "lr",
)
EVAL_DEPTH_MODE = getattr(kd_engine, "eval_depth_mode", "input")
if getattr(kd_engine, "trains_teacher_only", False) and config.KD.FREEZE_TEACHER:
    print("Teacher-only engine selected; overriding FREEZE_TEACHER so the teacher can be optimized.")
    for p in model.parameters():
        p.requires_grad = True

SCALAR_LOSS_KEYS = [key for key, value in dry_run_out.items()
                    if _is_scalar_loss_value(value) and key in CORE_LOSS_KEYS]

if "total" in SCALAR_LOSS_KEYS:
    SCALAR_LOSS_KEYS.remove("total")
    SCALAR_LOSS_KEYS.insert(0, "total")
else:
    SCALAR_LOSS_KEYS.insert(0, "total")

LOSS_KEY_TO_HEADER = {key: _display_name_for_loss(key) for key in SCALAR_LOSS_KEYS}
LOSS_HEADER_ORDER = [LOSS_KEY_TO_HEADER[key] for key in SCALAR_LOSS_KEYS]

print("Tracking losses:", ", ".join(LOSS_KEY_TO_HEADER.values()))

# ── 옵티마이저/스케줄러 ─────────────────────────────
def _teacher_checkpoint_payload(epoch: int, optimizer, metric_key: str, metric_value: float):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "teacher_state": teacher.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "use_amp": USE_AMP,
        "accum_steps": ACCUM_STEPS,
        metric_key: metric_value,
    }

    if hasattr(kd_engine, "get_decomposition_state") and callable(kd_engine.get_decomposition_state):
        decomp_state = kd_engine.get_decomposition_state()
        payload.update(decomp_state)
        payload["decomposition_state"] = decomp_state
        if "transfer_projector" not in payload:
            raise RuntimeError(
                "privileged teacher checkpoint must include transfer_projector, "
                "but the decomposition head has not been built."
            )

    return payload


optimizer_class = getattr(optim, config.TRAIN.OPTIMIZER["NAME"])

# [NEW] Param-group 분리: student vs KD-extra (projection/CSF 등)
# - student: lr=6e-5, wd=5e-3
# - kd-extra: lr=3e-4, wd=0
# NOTE: teacher는 freeze=False 이고 teacher CE를 쓰는 경우에만 optimizer에 포함
if getattr(kd_engine, "trains_teacher_only", False):
    student_params = list(model.parameters())
else:
    student_params = list(student.parameters())
kd_extra_params = list(kd_engine.get_extra_parameters())

teacher_params = []
if (
    (not getattr(kd_engine, "trains_teacher_only", False))
    and (not config.KD.FREEZE_TEACHER)
    and (config.KD.ENGINE_PARAMS.get("w_ce_teacher", 0.0) > 0.0)
):
    # teacher 와 student 가 같은 인스턴스면 중복 추가를 막는다
    if teacher is not student:
        teacher_params = list(teacher.parameters())

# Config로 제어 가능하도록 기본값 제공
pg = getattr(config.TRAIN, "PARAM_GROUPS", None)
if pg is None:
    pg = {
        "student": {"lr": 6e-5, "weight_decay": 5e-3},
        "kd_extra": {"lr": 3e-4, "weight_decay": 0.0},
        # teacher group은 필요할 때만 사용 (기본은 student와 동일하게 둠)
        "teacher": {"lr": 6e-5, "weight_decay": 5e-3},
    }

# AdamW 기타 하이퍼파라미터(betas, eps, amsgrad, fused 등)는 OPTIMIZER.PARAMS에서 가져오되
# group별 lr/wd는 여기에서 덮어씀
base_opt_params = dict(config.TRAIN.OPTIMIZER.get("PARAMS", {}))
base_opt_params.pop("lr", None)
base_opt_params.pop("weight_decay", None)

param_groups = []
param_groups.append(
    {"params": student_params, "lr": float(pg["student"]["lr"]), "weight_decay": float(pg["student"]["weight_decay"])}
)
if len(kd_extra_params) > 0:
    param_groups.append(
        {"params": kd_extra_params, "lr": float(pg["kd_extra"]["lr"]), "weight_decay": float(pg["kd_extra"]["weight_decay"])}
    )
if len(teacher_params) > 0:
    param_groups.append(
        {"params": teacher_params, "lr": float(pg["teacher"]["lr"]), "weight_decay": float(pg["teacher"]["weight_decay"])}
    )

optimizer = optimizer_class(param_groups, **base_opt_params)

ACCUM_STEPS = int(getattr(config.TRAIN, "ACCUM_STEPS", 1))
if ACCUM_STEPS < 1:
    raise ValueError(f"ACCUM_STEPS must be >= 1, got {ACCUM_STEPS}")

GRAD_CLIP_NORM = float(getattr(config.TRAIN, "GRAD_CLIP_NORM", 1.0))
USE_AMP = bool(getattr(config.TRAIN, "USE_AMP", True))
AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
scaler = GradScaler(enabled=bool(USE_AMP and AMP_DEVICE_TYPE == "cuda"))

warmup = None





# 1epoch당 학습 방법 설정 후 loss값 반환
def train_one_epoch_kd(
    kd_engine,
    loader,
    optimizer,
    device,
    epoch: int = 0,
    global_step_start: int = 0,
):
    kd_engine.train()
    if hasattr(kd_engine, "set_epoch"):
        try:
            kd_engine.set_epoch(epoch)
        except Exception:
            pass

    if not SCALAR_LOSS_KEYS:
        raise RuntimeError("No scalar losses registered from KD engine dry-run.")

    epoch_losses = {key: 0.0 for key in SCALAR_LOSS_KEYS}

    pbar = tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (imgs, masks) in enumerate(pbar):
        imgs = _move_to_device(imgs, device)
        masks = masks.to(device, non_blocking=True)

        # AMP forward (KD engine 내부에서 teacher/student forward 포함)
        amp_on = bool(USE_AMP and AMP_DEVICE_TYPE == "cuda")
        with autocast(AMP_DEVICE_TYPE, enabled=amp_on):
            out = kd_engine.compute_losses(imgs, masks, device)
            total_loss = out.get("total")
            if total_loss is None:
                raise KeyError("KD engine output does not contain 'total' loss.")
            if not isinstance(total_loss, torch.Tensor):
                raise TypeError("'total' loss must be a torch.Tensor for backpropagation.")

            loss_scaled = total_loss / ACCUM_STEPS

        if amp_on:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        do_update = ((batch_idx + 1) % ACCUM_STEPS == 0) or ((batch_idx + 1) == len(loader))
        if do_update:
            # grad clip: AMP면 unscale 후 clip
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                if amp_on:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(kd_engine.parameters(), max_norm=GRAD_CLIP_NORM)

            if amp_on:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        # 누적 loss (epoch 평균용)
        for key in epoch_losses:
            value = out.get(key)
            if value is None or not _is_scalar_loss_value(value):
                continue
            epoch_losses[key] += _loss_value_to_float(value)


        # tqdm postfix
        postfix = {}
        for key in SCALAR_LOSS_KEYS:
            value = out.get(key)
            if value is None or not _is_scalar_loss_value(value):
                continue
            postfix_label = LOSS_KEY_TO_HEADER.get(key, key)
            postfix[postfix_label] = f"{_loss_value_to_float(value):.3f}"
        if postfix:
            pbar.set_postfix(postfix)

    num_batches = len(loader)
    avg_losses = {key: val / num_batches for key, val in epoch_losses.items()}
    return avg_losses


# ── (변경) 검증(학생 기준) ─────────────────────────────────
def validate_student(
    student_model,
    loader,
    criterion,
    eval_view: str = "lr",
    eval_depth_mode: str = "input",
):
    student_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Validation"):
            imgs = _move_to_device(imgs, device)
            masks = masks.to(device, non_blocking=True)
            preds = _forward_eval_model(
                student_model,
                imgs,
                eval_view=eval_view,
                eval_depth_mode=eval_depth_mode,
            )  # logits

            if preds.shape[-2:] != masks.shape[-2:]:
                preds = F.interpolate(
                    preds,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            total_loss += criterion(preds, masks).item()
    return total_loss / len(loader)




def write_summary(init=False, best_epoch=None, best_miou=None):
    with open(config.GENERAL.SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("=== Training Configuration ===\n")
        f.write(f"Dataset path : {config.DATA.DATA_DIR}\n")
        og = optimizer.param_groups[0]
        f.write(f"Student Model: {student.__class__.__name__}  (source: {config.KD.STUDENT_NAME})\n")
        f.write(f"Teacher Model: {teacher.__class__.__name__}  (source: {config.KD.TEACHER_NAME})\n")
        f.write(f"Teacher is Student: {teacher is student}\n\n")
        f.write(f"Teacher Freeze: {config.KD.FREEZE_TEACHER}\n")
        f.write(f"Main Eval View : {EVAL_VIEW}\n")
        f.write(f"Eval Depth Mode: {EVAL_DEPTH_MODE}\n")
        f.write(f"Optimizer     : {optimizer.__class__.__name__}\n")
        f.write("  --- Param Groups ---\n")
        for gi, g in enumerate(optimizer.param_groups):
            f.write(f"  group[{gi}] lr={g.get('lr')} wd={g.get('weight_decay')} n_params={len(g.get('params', []))}\n")
        f.write(f"AMP           : {USE_AMP} (device_type={AMP_DEVICE_TYPE})\n")
        f.write(f"Accum steps   : {ACCUM_STEPS}\n")
        f.write(f"Batch size    : {config.DATA.BATCH_SIZE}\n\n")
        f.write("=== Knowledge Distillation Configuration ===\n")
        f.write(f"Engine NAME        : {config.KD.ENGINE_NAME}\n")
        f.write(f"Teacher Source CKPT: {config.TEACHER_CKPT}\n\n")

        engine_name = config.KD.ENGINE_NAME
        current_engine_params = config.KD.ALL_ENGINE_PARAMS.get(engine_name, {})
        f.write(f"--- Parameters for '{engine_name}' engine ---\n")
        if not current_engine_params:
            f.write("No parameters found for this engine.\n")
        else:
            for key, value in current_engine_params.items():
                f.write(f"{key:<25} : {value}\n")
        f.write("\n")

        if init:
            f.write("=== Best Model (to be updated) ===\n")
            f.write("epoch     : N/A\nbest_val_mIoU : N/A\n\n")
        else:
            f.write("=== Best Model ===\n")
            f.write(f"epoch     : {best_epoch}\n")
            f.write(f"best_val_mIoU : {best_miou:.4f}\n\n")


def write_timing(start_dt, end_dt, path=config.GENERAL.SUMMARY_TXT):
    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    with open(path, "a", encoding="utf-8") as f:
        f.write("=== Timing ===\n")
        f.write(f"Start : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)\n\n")



# 학습 진행 및 잘 되고있나 성능평가
def run_training(num_epochs):
    write_summary(init=True)
    start_dt = datetime.now()
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")

    best_val_miou = -float("inf")
    best_val_epoch = 0

    best_val_ckpt  = config.GENERAL.BASE_DIR / "best_model_val.pth"
    final_best_ckpt = config.GENERAL.BASE_DIR / "best_model.pth"

    # 마지막 50 epoch만 테스트 세트 평가에 사용 (총 epoch이 50보다 작으면 전체 평가)

    # ────────────────────────────────────────────────────────────
    # [PATCH 3] CSV 헤더에 "HR Test mIoU" 추가
    # ────────────────────────────────────────────────────────────
    log_csv_path = config.GENERAL.LOG_DIR / "training_log.csv"
    loss_headers = LOSS_HEADER_ORDER if LOSS_HEADER_ORDER else ["Total Loss"]
    csv_headers = [
        "Epoch", *loss_headers, "Val Loss", "Val mIoU", "Pixel Acc",
        "LR",
    ]
    for i in range(config.DATA.NUM_CLASSES):
        csv_headers.append(f"IoU_{config.DATA.CLASS_NAMES[i]}")

    if log_csv_path.exists():
        try:
            existing_cols = list(pd.read_csv(log_csv_path, nrows=0).columns)
            if len(existing_cols) > 0:
                csv_headers = existing_cols
        except Exception:
            pass
    else:
        pd.DataFrame(columns=csv_headers).to_csv(log_csv_path, index=False)

    train_losses, val_losses = [], []
    loss_class = getattr(nn, config.TRAIN.LOSS_FN["NAME"])
    criterion = loss_class(**config.TRAIN.LOSS_FN["PARAMS"])

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        tr_losses_dict = train_one_epoch_kd(
            kd_engine,
            data_loader.train_loader,
            optimizer,
            device,
            epoch=epoch,
            global_step_start=global_step,
        )
        tr_loss = tr_losses_dict["total"]
        global_step += len(data_loader.train_loader)

        vl_loss = validate_student(
            model,
            data_loader.val_loader,
            criterion,
            eval_view=EVAL_VIEW,
            eval_depth_mode=EVAL_DEPTH_MODE,
        )
        metrics = evaluate.evaluate_all(
            model,
            data_loader.val_loader,
            device,
            eval_view=EVAL_VIEW,
            eval_depth_mode=EVAL_DEPTH_MODE,
        )
        miou = metrics["mIoU"]
        pa = metrics["PixelAcc"]

        # ------------------------------------------------------------
        # Best Val mIoU checkpoint 저장 (항상 평가 가능)
        # ------------------------------------------------------------
        if miou > best_val_miou:
            best_val_miou = miou
            best_val_epoch = epoch
            torch.save(
                _teacher_checkpoint_payload(
                    epoch=epoch,
                    optimizer=optimizer,
                    metric_key="best_val_mIoU",
                    metric_value=best_val_miou,
                ),
                best_val_ckpt,
            )
            print(f"New best val_mIoU at epoch {epoch}: {miou:.4f} -> {best_val_ckpt}")
            write_summary(init=False, best_epoch=best_val_epoch, best_miou=best_val_miou)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(
            f"[{epoch}/{num_epochs}] "
            f"train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}, "
            f"val_mIoU={miou:.4f}, PA={pa:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]


        # CSV 파일에 성능 지표 기록
        log_data = {"Epoch": epoch}
        for key in SCALAR_LOSS_KEYS:
            header_name = LOSS_KEY_TO_HEADER.get(key, key)
            log_data[header_name] = tr_losses_dict.get(key, float("nan"))

        log_data.update(
            {
                "Val Loss": vl_loss,
                "Val mIoU": miou,
                "Pixel Acc": pa,
                "LR": current_lr,
            }
        )

        per_cls_iou = metrics["per_class_iou"]
        for i in range(config.DATA.NUM_CLASSES):
            log_data[f"IoU_{config.DATA.CLASS_NAMES[i]}"] = float(per_cls_iou[i])

        df_new_row = pd.DataFrame([log_data]).reindex(columns=csv_headers)
        df_new_row.to_csv(log_csv_path, mode="a", header=False, index=False)

    end_dt = datetime.now()
    write_timing(start_dt, end_dt, config.GENERAL.SUMMARY_TXT)

    elapsed = end_dt - start_dt
    total_sec = int(elapsed.total_seconds())
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60

    print("\nTraining complete.")
    print(f"Started at : {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Finished at: {end_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Total time : {hh:02d}:{mm:02d}:{ss:02d} (H:M:S)")
    print(f"Best epoch (val mIoU): {best_val_epoch}, Best val_mIoU: {best_val_miou:.4f}")

    # best model 은 validation mIoU 단독 기준으로 선정한다.
    shutil.copy2(best_val_ckpt, final_best_ckpt)
    print(f"Final best model (val-selected, epoch={best_val_epoch}) -> {final_best_ckpt}")

    ckpt = torch.load(final_best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    m = evaluate.evaluate_all(
        model,
        data_loader.test_loader,
        device,
        eval_view=EVAL_VIEW,
        eval_depth_mode=EVAL_DEPTH_MODE,
    )
    print(f"Test mIoU: {m['mIoU']:.4f}, Test Pixel Acc: {m['PixelAcc']:.4f}")

    return final_best_ckpt


if __name__ == "__main__":
    run_training(config.TRAIN.EPOCHS)
