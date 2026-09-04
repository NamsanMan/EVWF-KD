import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import shutil
import math
import inspect
import re
import numpy as np
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
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# 보기 싫은 로그 숨김
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

LOSS_KEY_DISPLAY_OVERRIDES = {
    "total": "Total Loss",
    "ce_student": "CE Student Loss",
    "sgsc_dec": "Decoder KD Loss",
    "sgsc_dec_raw": "Decoder KD Raw",
    "reachable_geo_stage_sgsc": "Encoder KD Loss",
    "reachable_geo_stage_sgsc_raw": "Encoder KD Raw",
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


# model 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher = create_model(config.KD.TEACHER_NAME).to(device)
student = create_model(config.KD.STUDENT_NAME).to(device)
model = student


def _legacy_segformer_key_to_hf(key: str) -> str:
    rules = [
        (
            r"^model\.segformer\.stages\.(\d+)\.patch_embeddings\.",
            r"model.segformer.encoder.patch_embeddings.\1.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.layernorm_before\.",
            r"model.segformer.encoder.block.\1.\2.layer_norm_1.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.layernorm_after\.",
            r"model.segformer.encoder.block.\1.\2.layer_norm_2.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.attention\.q_proj\.",
            r"model.segformer.encoder.block.\1.\2.attention.self.query.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.attention\.k_proj\.",
            r"model.segformer.encoder.block.\1.\2.attention.self.key.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.attention\.v_proj\.",
            r"model.segformer.encoder.block.\1.\2.attention.self.value.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.attention\.o_proj\.",
            r"model.segformer.encoder.block.\1.\2.attention.output.dense.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.attention\.sequence_reduction\.sequence_reduction\.",
            r"model.segformer.encoder.block.\1.\2.attention.self.sr.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.attention\.sequence_reduction\.layer_norm\.",
            r"model.segformer.encoder.block.\1.\2.attention.self.layer_norm.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc1\.",
            r"model.segformer.encoder.block.\1.\2.mlp.dense1.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc2\.",
            r"model.segformer.encoder.block.\1.\2.mlp.dense2.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.blocks\.(\d+)\.mlp\.dwconv\.",
            r"model.segformer.encoder.block.\1.\2.mlp.dwconv.",
        ),
        (
            r"^model\.segformer\.stages\.(\d+)\.layer_norm\.",
            r"model.segformer.encoder.layer_norm.\1.",
        ),
        (
            r"^model\.decode_head\.linear_projections\.(\d+)\.proj\.",
            r"model.decode_head.linear_c.\1.proj.",
        ),
    ]
    for pattern, repl in rules:
        key = re.sub(pattern, repl, key)
    return key


def _prepare_teacher_state_dict_for_load(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        removed_module_prefix = True
    else:
        removed_module_prefix = False

    has_legacy_keys = any(
        ".segformer.stages." in key or ".decode_head.linear_projections." in key
        for key in sd.keys()
    )
    if not has_legacy_keys:
        return sd, {
            "module_prefix_removed": removed_module_prefix,
            "legacy_remapped": False,
            "changed_keys": 0,
            "total_keys": len(sd),
        }

    converted = {_legacy_segformer_key_to_hf(key): value for key, value in sd.items()}
    changed = sum(
        1 for key in sd.keys() if _legacy_segformer_key_to_hf(key) != key
    )
    return converted, {
        "module_prefix_removed": removed_module_prefix,
        "legacy_remapped": True,
        "changed_keys": changed,
        "total_keys": len(sd),
    }


TEACHER_CKPT_LOAD_INFO = {
    "original_ckpt": str(getattr(config, "TEACHER_CKPT_ORIGINAL", config.TEACHER_CKPT)),
    "load_ckpt": str(config.TEACHER_CKPT),
    "loaded": False,
    "exists": False,
    "strict": True,
    "module_prefix_removed": False,
    "legacy_remapped": bool(getattr(config, "TEACHER_CKPT_REMAP_APPLIED_UPSTREAM", False)),
    "legacy_remapped_in_train": False,
    "legacy_remapped_upstream": bool(getattr(config, "TEACHER_CKPT_REMAP_APPLIED_UPSTREAM", False)),
    "remap_source": str(getattr(config, "TEACHER_CKPT_REMAP_SOURCE_UPSTREAM", "")) or "none",
    "changed_keys": 0,
    "total_keys": 0,
    "upstream_detail": str(getattr(config, "TEACHER_CKPT_REMAP_DETAIL_UPSTREAM", "")),
}


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
        original_ckpt_path = Path(getattr(config, "TEACHER_CKPT_ORIGINAL", ckpt_path))
        upstream_remapped = bool(getattr(config, "TEACHER_CKPT_REMAP_APPLIED_UPSTREAM", False))
        upstream_source = str(getattr(config, "TEACHER_CKPT_REMAP_SOURCE_UPSTREAM", "")) or "sweep"
        upstream_detail = str(getattr(config, "TEACHER_CKPT_REMAP_DETAIL_UPSTREAM", ""))
        TEACHER_CKPT_LOAD_INFO.update(
            {
                "original_ckpt": str(original_ckpt_path),
                "load_ckpt": str(ckpt_path),
                "legacy_remapped_upstream": upstream_remapped,
                "upstream_detail": upstream_detail,
            }
        )
        if ckpt_path.exists():
            TEACHER_CKPT_LOAD_INFO["exists"] = True
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            # --- STRICT teacher checkpoint validation ---
            sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

            # Teacher checkpoint key compatibility cleanup.
            sd, key_info = _prepare_teacher_state_dict_for_load(sd)
            remapped = bool(key_info["legacy_remapped"] or upstream_remapped)
            if key_info["legacy_remapped"]:
                remap_source = "train_kd"
            elif upstream_remapped:
                remap_source = upstream_source
            else:
                remap_source = "none"
            TEACHER_CKPT_LOAD_INFO.update(
                {
                    "module_prefix_removed": bool(key_info["module_prefix_removed"]),
                    "legacy_remapped": remapped,
                    "legacy_remapped_in_train": bool(key_info["legacy_remapped"]),
                    "remap_source": remap_source,
                    "changed_keys": int(key_info["changed_keys"]),
                    "total_keys": int(key_info["total_keys"]),
                }
            )
            if key_info["legacy_remapped"]:
                print(
                    "[INFO] Teacher checkpoint key remap: applied "
                    f"(legacy SegFormer -> HF, {key_info['changed_keys']}/"
                    f"{key_info['total_keys']} keys changed)."
                )
            else:
                print("[INFO] Teacher checkpoint key remap: not needed.")
            if key_info["module_prefix_removed"]:
                print("[INFO] Teacher checkpoint key cleanup: removed 'module.' prefix.")

            ret = teacher.load_state_dict(sd, strict=True)
            TEACHER_CKPT_LOAD_INFO["loaded"] = True
            if original_ckpt_path != ckpt_path:
                print(f"[INFO] Teacher checkpoint original path: {original_ckpt_path}")
                print(f"[INFO] Teacher checkpoint effective load path: {ckpt_path}")
            print(f"[INFO] Teacher checkpoint key remap recorded: {remapped} (source={remap_source})")
            ckpt_path = original_ckpt_path
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

# ── Gavish-Donoho calibration (해당 engine만) ──────
if hasattr(kd_engine, "calibrate") and callable(kd_engine.calibrate):
    kd_engine.calibrate(data_loader.train_loader, device=device)

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


def _forward_eval_model(model, imgs, eval_view: str = "lr"):
    """
    기존 RGB-only 모델과 depth-fusion 모델을 모두 지원.

    imgs 형태:
      - Tensor
      - (x_lr, x_hr)
      - (x_lr, depth)            # depth-aware wrapper 데이터셋
      - (x_lr, x_hr, depth)
      - 더 긴 tuple/list여도 앞 3개만 사용

    depth-aware 모델이면 depth 인자를 전달하고, 아니면 depth는 버리고 RGB-only 경로로 forward.
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
            return model(x, depth=depth)
        return model(x)

    if len(imgs) == 2:
        a, b = imgs
        # depth-aware 데이터셋: (x_lr, depth)의 경우 depth는 (B,1,H,W)
        if use_depth and torch.is_tensor(b) and b.dim() == 4 and b.size(1) == 1:
            return model(a, depth=b)
        if use_depth and torch.is_tensor(b) and b.dim() == 3 and b.size(0) == 1:
            return model(a, depth=b.unsqueeze(0))
        # 그 외: (x_lr, x_hr) 형태
        x = a if eval_view.lower() == "lr" else b
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
optimizer_class = getattr(optim, config.TRAIN.OPTIMIZER["NAME"])

# [NEW] Param-group 분리: student vs KD-extra (projection/CSF 등)
# - student: lr=6e-5, wd=5e-3
# - kd-extra: lr=3e-4, wd=0
# NOTE: teacher는 freeze=False 이고 teacher CE를 쓰는 경우에만 optimizer에 포함
student_params = list(student.parameters())
kd_extra_params = list(kd_engine.get_extra_parameters())

teacher_params = []
if (not config.KD.FREEZE_TEACHER) and (config.KD.ENGINE_PARAMS.get("w_ce_teacher", 0.0) > 0.0):
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
def validate_student(student_model, loader, criterion):
    """
    student 기준으로 validation loss를 계산.
    - imgs는 Tensor 혹은 (x_lr, ...), (x_lr, depth), (x_lr, x_hr, depth) 등 tuple일 수 있음.
    - student가 depth-aware라면 depth를 넘기고, 아니면 무시 (RGB-only 경로).
    - pred와 mask 해상도가 다르면 pred를 mask 해상도로 업샘플 후 loss 계산.
    """
    student_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, ascii=True, dynamic_ncols=True, leave=True, desc="Validation"):
            imgs = _move_to_device(imgs, device)
            masks = masks.to(device, non_blocking=True)
            preds = _forward_eval_model(student_model, imgs, eval_view="lr")  # logits

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
        f.write(f"Teacher Model: {teacher.__class__.__name__}  (source: {config.KD.TEACHER_NAME})\n\n")
        f.write(f"Teacher Freeze: {config.KD.FREEZE_TEACHER}\n")
        f.write("Main Eval View : lr\n")
        f.write(f"Optimizer     : {optimizer.__class__.__name__}\n")
        f.write("  --- Param Groups ---\n")
        for gi, g in enumerate(optimizer.param_groups):
            f.write(f"  group[{gi}] lr={g.get('lr')} wd={g.get('weight_decay')} n_params={len(g.get('params', []))}\n")
        f.write(f"AMP           : {USE_AMP} (device_type={AMP_DEVICE_TYPE})\n")
        f.write(f"Accum steps   : {ACCUM_STEPS}\n")
        f.write(f"Batch size    : {config.DATA.BATCH_SIZE}\n\n")
        f.write("=== Knowledge Distillation Configuration ===\n")
        f.write(f"Engine NAME        : {config.KD.ENGINE_NAME}\n")
        f.write(f"Teacher Source CKPT: {TEACHER_CKPT_LOAD_INFO['original_ckpt']}\n")
        f.write(f"Teacher Load CKPT  : {TEACHER_CKPT_LOAD_INFO['load_ckpt']}\n")
        f.write(f"Teacher CKPT Loaded: {TEACHER_CKPT_LOAD_INFO['loaded']}\n")
        f.write(
            "Teacher Key Remap  : "
            f"{TEACHER_CKPT_LOAD_INFO['legacy_remapped']} "
            f"(source={TEACHER_CKPT_LOAD_INFO['remap_source']})\n"
        )
        f.write(
            "Teacher Key Details: "
            f"changed_in_train={TEACHER_CKPT_LOAD_INFO['changed_keys']}/"
            f"{TEACHER_CKPT_LOAD_INFO['total_keys']}, "
            f"module_prefix_removed={TEACHER_CKPT_LOAD_INFO['module_prefix_removed']}, "
            f"upstream_remap={TEACHER_CKPT_LOAD_INFO['legacy_remapped_upstream']}"
        )
        if TEACHER_CKPT_LOAD_INFO["upstream_detail"]:
            f.write(f", upstream_detail={TEACHER_CKPT_LOAD_INFO['upstream_detail']}")
        f.write("\n\n")

        engine_name = config.KD.ENGINE_NAME
        current_engine_params = config.KD.ALL_ENGINE_PARAMS.get(
            engine_name,
            getattr(config.KD, "ENGINE_PARAMS", {}),
        )
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


# ────────────────────────────────────────────────────────────────
# HR test evaluation helper (optional, parity with train_teacher.py)
# ────────────────────────────────────────────────────────────────


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

    # CSV 로그 파일 경로 설정 및 헤더 생성
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
            existing_df = pd.read_csv(log_csv_path)
            existing_cols = list(existing_df.columns)
            if len(existing_cols) > 0:
                csv_headers = existing_cols + [
                    col for col in csv_headers if col not in existing_cols
                ]
                existing_df.reindex(columns=csv_headers).to_csv(log_csv_path, index=False)
        except Exception:
            pd.DataFrame(columns=csv_headers).to_csv(log_csv_path, index=False)
    else:
        pd.DataFrame(columns=csv_headers).to_csv(log_csv_path, index=False)

    train_losses, val_losses = [], []
    loss_class = getattr(nn, config.TRAIN.LOSS_FN["NAME"])
    criterion = loss_class(**config.TRAIN.LOSS_FN["PARAMS"])

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        # Engines with an epoch-aware schedule (e.g. DPCLDEngine.w_kd decay)
        # get notified here. Engines without this hook are unaffected.
        if hasattr(kd_engine, "set_current_epoch"):
            kd_engine.set_current_epoch(epoch)

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

        vl_loss = validate_student(student, data_loader.val_loader, criterion)
        metrics = evaluate.evaluate_all(model, data_loader.val_loader, device, eval_view="lr")
        miou = metrics["mIoU"]
        pa = metrics["PixelAcc"]

        # ------------------------------------------------------------
        # NEW: Best Val mIoU checkpoint 저장 (항상 평가 가능)
        # ------------------------------------------------------------
        if miou > best_val_miou:
            best_val_miou = miou
            best_val_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": student.state_dict(),
                    "teacher_state": teacher.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "use_amp": USE_AMP,
                    "accum_steps": ACCUM_STEPS,
                    "best_val_mIoU": best_val_miou,
                },
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
        if "debug_dec_sgsc_to_depth" in tr_losses_dict:
            print(
                "[KD-Debug] "
                f"dec_sgsc depth/hr0/lr0="
                f"{tr_losses_dict.get('debug_dec_sgsc_to_depth', float('nan')):.4f}/"
                f"{tr_losses_dict.get('debug_dec_sgsc_to_hr_zero', float('nan')):.4f}/"
                f"{tr_losses_dict.get('debug_dec_sgsc_to_lr_zero', float('nan')):.4f}, "
                f"dec_rel_gap depth-hr0/depth-lr0="
                f"{tr_losses_dict.get('debug_dec_depth_vs_hr_zero_rel_l2', float('nan')):.4f}/"
                f"{tr_losses_dict.get('debug_dec_depth_vs_lr_zero_rel_l2', float('nan')):.4f}"
            )
            print(
                "[KD-Debug] "
                f"enc_sgsc depth/hr0/lr0="
                f"{tr_losses_dict.get('debug_stage_sgsc_to_depth_mean', float('nan')):.4f}/"
                f"{tr_losses_dict.get('debug_stage_sgsc_to_hr_zero_mean', float('nan')):.4f}/"
                f"{tr_losses_dict.get('debug_stage_sgsc_to_lr_zero_mean', float('nan')):.4f}, "
                f"enc_rel_gap depth-hr0/depth-lr0="
                f"{tr_losses_dict.get('debug_stage_depth_vs_hr_zero_rel_l2_mean', float('nan')):.4f}/"
                f"{tr_losses_dict.get('debug_stage_depth_vs_lr_zero_rel_l2_mean', float('nan')):.4f}"
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
    # test set 은 아래에서 최종 성능을 한 번 보고할 때만 사용한다.
    shutil.copy2(best_val_ckpt, final_best_ckpt)
    print(f"Final best model (val-selected, epoch={best_val_epoch}) -> {final_best_ckpt}")

    ckpt = torch.load(final_best_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state"])
    student.eval()
    m = evaluate.evaluate_all(student, data_loader.test_loader, device, eval_view="lr")
    print(f"Test mIoU: {m['mIoU']:.4f}, Test Pixel Acc: {m['PixelAcc']:.4f}")

    return final_best_ckpt


if __name__ == "__main__":
    run_training(config.TRAIN.EPOCHS)
