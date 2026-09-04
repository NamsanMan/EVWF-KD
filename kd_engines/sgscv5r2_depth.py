"""
Spectral-Guided Spatial Correlation KD Engine — Depth variant
SGSCv5r2-depth

============================================================================
기반: kd_engines/sgscv5r2.py (SGSCEngine)

변경점 (핵심):
----------------------------------------------------------------------------
원본 sgscv5r2.py는 teacher HR stream feature로부터 basis(U)를 계산하고,
student LR feature와 teacher HR feature를 모두 U로 projection했다.

본 sgscv5r2_depth.py는 depth로 학습된 teacher를 전제로 다음과 같이 변경:

  1. Teacher는 (HR image, depth)을 입력으로 받아 forward.
     (HR+depth stream의 feature를 teacher 측 기준 신호로 사용.
      data_loader에서 x_lr, x_hr, depth 모두 동일한 (H,W)=INPUT_RESOLUTION
      으로 리사이즈되어 들어오므로 공간적 불일치는 없음.)
  2. Basis(U)는 teacher의 HR+depth stream feature의 batch covariance에서
     top-k eigenvectors로 계산.
  3. Teacher의 HR+depth feature를 U에 projection하여 Z_t를 구함.
  4. Student는 LR image만 입력으로 받고, feature를 동일한 U에 projection
     하여 Z_s를 구함.
  5. 나머지 (per-channel spatial PCC, z-score, etc.)은 sgscv5r2.py와 동일.

============================================================================
Input contract:
----------------------------------------------------------------------------
`compute_losses(imgs, masks, device)` 의 `imgs` 는 다음 형식을 지원.

  - (x_lr, x_hr, depth)            : 표준 3-tuple (권장)
  - (x_lr, x_hr)                   : 2-tuple (depth 없음 → zero-depth fallback)
  - (x_lr, depth)                  : 2-tuple (HR 없음 → x_lr을 HR 대용으로 사용)
  - (x_lr,) 또는 x_lr (tensor)    : HR/depth 모두 없음 → 모두 fallback

Student는 항상 `x_lr`만 forward.
Teacher는 depth-aware forward (`depth=...` 키워드) 를 호출하며, HR 이미지가
없으면 x_lr을, depth가 없으면 zero-depth를 fallback으로 사용한다.

============================================================================
"""

from __future__ import annotations

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Iterable, Optional, Tuple

import config
from .base_engine import BaseKDEngine


# =============================================================================
# Utility functions
# =============================================================================

def _autocast_off(device: torch.device):
    """AMP autocast를 명시적으로 비활성화하는 context manager."""
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=False)

    class _Dummy:
        def __enter__(self):
            return None
        def __exit__(self, *args):
            return False

    return _Dummy()


def _safe_groupnorm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


def _interpolate_antialias(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size_hw:
        return x
    return F.interpolate(
        x, size=size_hw, mode="bilinear", align_corners=False, antialias=True
    )


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _supports_depth_forward(model: nn.Module) -> bool:
    """
    model.forward 시그니처에 `depth` 인자가 있는지 확인.
    """
    m = _unwrap_model(model)
    try:
        sig = inspect.signature(m.forward)
        return "depth" in sig.parameters
    except Exception:
        return False


# =============================================================================
# Core: Spectral-Guided Spatial PCC Loss  (identical to sgscv5r2.py)
# =============================================================================

class SpectralGuidedSpatialPCC(nn.Module):
    """
    단일 stage에 대한 SGSC loss.

    1. Teacher batch covariance → eigh → top-k eigenvectors
    2. Projection Z = U^T · (F - μ)
    3. Per-channel spatial PCC: L = (1/k) Σ 2(1 - ρ_k)

    Args:
        k:   projection 차원 (고정)
        eps: 수치 안정성
    """

    def __init__(self, k: int = 64, eps: float = 1e-6):
        super().__init__()
        self.k = int(k)
        self.eps = eps

    def _estimate_basis(self, t_centered: torch.Tensor) -> torch.Tensor:
        """
        Batch covariance → eigh → top-k eigenvectors.

        Args:
            t_centered: (B, C, N) centered teacher features
        Returns:
            U: (C, k) orthonormal basis, top-k by eigenvalue
        """
        device = t_centered.device

        with _autocast_off(device):
            t_c = t_centered.float()
            B, C, N = t_c.shape
            k = min(self.k, C)

            # Batch covariance (C, C) — no shrinkage
            cov = torch.einsum("bcn,bdn->cd", t_c, t_c) / (float(B * N) + self.eps)

            # Full eigendecomposition (ascending order)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            # Top-k eigenvectors (largest eigenvalues = last k columns)
            U = eigenvectors[:, -k:]  # (C, k)
            return U

    def _spatial_pcc(self, Z_s: torch.Tensor, Z_t: torch.Tensor) -> torch.Tensor:
        """
        Per-channel spatial PCC loss.

        Z는 centering된 feature의 linear projection이므로 spatial mean = 0.
        따라서 std = RMS이고, z-score = Z / RMS(Z).

        MSE(z_s, z_t) = 2(1 - ρ)  (Pearson distance의 2배)

        Args:
            Z_s, Z_t: (B, k, H, W)
        Returns:
            loss: scalar
        """
        B, K, H, W = Z_s.shape
        z_s = Z_s.view(B, K, -1)  # (B, K, N)
        z_t = Z_t.view(B, K, -1)

        # RMS per channel (= std since mean=0)
        rms_s = z_s.pow(2).mean(dim=2, keepdim=True).sqrt()  # (B, K, 1)
        rms_t = z_t.pow(2).mean(dim=2, keepdim=True).sqrt()

        # z-score = Z / RMS
        z_s_n = z_s / (rms_s + self.eps)
        z_t_n = z_t / (rms_t + self.eps)

        return F.mse_loss(z_s_n, z_t_n)

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_s: (B, C, H, W) student feature (channel-aligned, LR-only)
            f_t: (B, C, H, W) teacher feature (HR+depth, downsampled)
        Returns:
            loss: scalar

        Note:
            Basis U는 f_t (teacher HR+depth feature) 에서만 계산되며
            student와 teacher 모두 동일한 U에 projection된다.
        """
        B, C, H, W = f_s.shape
        N = H * W
        device = f_s.device

        with _autocast_off(device):
            s_flat = f_s.float().view(B, C, N)
            t_flat = f_t.float().view(B, C, N)

            # Spatial centering
            s_centered = s_flat - s_flat.mean(dim=2, keepdim=True)
            t_centered = t_flat - t_flat.mean(dim=2, keepdim=True)

            # PCA basis from teacher LR+depth feature (no grad)
            with torch.no_grad():
                U = self._estimate_basis(t_centered).detach()  # (C, k)

            k = U.shape[1]

            # Projection — 양쪽 모두 teacher LR+depth basis에 투영
            Z_s = torch.einsum("ck,bcn->bkn", U, s_centered).view(B, k, H, W)
            Z_t = torch.einsum("ck,bcn->bkn", U, t_centered).view(B, k, H, W)

            loss = self._spatial_pcc(Z_s, Z_t)

        return loss


# =============================================================================
# KD Engine (Depth variant)
# =============================================================================

class SGSCDepthEngine(BaseKDEngine):
    """
    SGSC KD Engine — v5r2 depth variant.

    Teacher가 (HR image, depth) 를 입력으로 받아 HR+depth stream feature를
    생성하고, 이 feature로부터 basis(U)를 계산하여 student LR feature와
    teacher HR+depth feature를 모두 U에 projection하여 PCC를 계산한다.
    (x_lr, x_hr, depth 모두 data_loader 단계에서 동일한 (H,W)=INPUT_RESOLUTION
    으로 리사이즈되므로 공간 정렬을 위한 추가 interpolate는 필요하지 않다.)

    Config:
        "sgsc_depth": {
            "w_ce_student": 1.0,
            "w_kd": 0.5,

            "k": 64,

            "apply_stages": (1, 2, 3),
            "stage_weights": {1: 0.5, 2: 1.0, 3: 1.25},

            "student_channels": [32, 64, 160, 256],
            "teacher_channels": [64, 128, 320, 512],

            "ignore_index": 11,
        },
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student)

        # --- Loss weights ---
        self.w_ce_student = float(kwargs.get("w_ce_student", 1.0))
        self.w_kd = float(kwargs.get("w_kd", 0.5))

        # --- Subspace config ---
        self.k = int(kwargs.get("k", 64))

        # --- Stage config ---
        self.apply_stages = tuple(kwargs.get("apply_stages", (1, 2, 3)))
        self.stage_weights = kwargs.get("stage_weights", {1: 0.5, 2: 1.0, 3: 1.25})

        # --- Channel alignment projectors ---
        student_ch = kwargs.get("student_channels", [32, 64, 160, 256])
        teacher_ch = kwargs.get("teacher_channels", [64, 128, 320, 512])

        self.projectors = nn.ModuleList()
        for i in range(4):
            if i in self.apply_stages:
                C_out = int(teacher_ch[i])
                self.projectors.append(
                    nn.Sequential(
                        nn.Conv2d(int(student_ch[i]), C_out, kernel_size=1, bias=True),
                        _safe_groupnorm(C_out, max_groups=32),
                    )
                )
            else:
                self.projectors.append(nn.Identity())

        self._extra_params = list(self.projectors.parameters())

        self.ignore_index = int(
            kwargs.get("ignore_index", getattr(config.DATA, "IGNORE_INDEX", 255))
        )

        # --- SGSC loss module ---
        self.sgsc_loss = SpectralGuidedSpatialPCC(k=self.k)

        # --- CE loss ---
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # --- Force patch embed extraction ---
        if hasattr(self.teacher, "set_force_patch_embeds"):
            self.teacher.set_force_patch_embeds(True)
        if hasattr(self.student, "set_force_patch_embeds"):
            self.student.set_force_patch_embeds(True)

        # --- Depth-aware forward detection (cached once) ---
        self._teacher_supports_depth = _supports_depth_forward(self.teacher)
        if not self._teacher_supports_depth:
            # depth 입력을 무시하게 되므로 경고. (강제 실패하지는 않음)
            print(
                "[SGSCDepthEngine] WARNING: teacher.forward does not accept `depth=`. "
                "Depth input will be ignored; basis는 teacher HR-only stream으로 계산됩니다."
            )

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        return self._extra_params

    # -------------------------------------------------------------------------
    # Input unpack helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _unpack_inputs(
        imgs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return (x_lr, x_hr_or_None, depth_or_None).

        data_loader 기준 지원 형식:
          - tensor                    -> (tensor, None, None)
          - (x_lr,)                   -> (x_lr, None, None)
          - (x_lr, x_hr)              -> (x_lr, x_hr, None)
          - (x_lr, depth)             -> (x_lr, None, depth)
              depth는 (B,1,H,W) 또는 (1,H,W)로 구분
          - (x_lr, x_hr, depth, ...)  -> (x_lr, x_hr, depth)
        """
        if not isinstance(imgs, (tuple, list)):
            return imgs, None, None

        if len(imgs) == 0:
            raise RuntimeError("Empty imgs received by SGSCDepthEngine.")

        if len(imgs) == 1:
            return imgs[0], None, None

        if len(imgs) >= 3:
            # (x_lr, x_hr, depth, ...) 표준 형식
            return imgs[0], imgs[1], imgs[2]

        # len == 2 — depth / HR 판별
        a, b = imgs[0], imgs[1]
        # depth는 단일 채널 텐서로 가정. (B,1,H,W) 또는 (1,H,W)
        if torch.is_tensor(b) and b.dim() == 4 and b.size(1) == 1:
            return a, None, b
        if torch.is_tensor(b) and b.dim() == 3 and b.size(0) == 1:
            return a, None, b.unsqueeze(0)
        # RGB-like (B,3,H,W) 이면 HR stream 으로 해석
        return a, b, None

    def _zero_depth_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), 1, x.size(2), x.size(3),
                           dtype=x.dtype, device=x.device)

    # -------------------------------------------------------------------------
    # compute_losses
    # -------------------------------------------------------------------------

    def compute_losses(
        self, imgs: Any, masks: torch.Tensor, device
    ) -> Dict[str, Any]:
        # --- Input unpack ---
        x_lr, x_hr, depth = self._unpack_inputs(imgs)

        # HR 이미지가 없으면 x_lr을 HR 입력 자리로 fallback (공간 크기는 동일)
        x_teacher_rgb = x_hr if x_hr is not None else x_lr

        # --- Masks shape ---
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        if masks.dim() != 3:
            raise RuntimeError(f"masks must be (B,H,W), got {tuple(masks.shape)}")

        # --- Student forward (LR only) ---
        s_out = self.student(x_lr, is_feat=True)
        if not (isinstance(s_out, (tuple, list)) and len(s_out) >= 2):
            raise RuntimeError(f"Student wrapper output mismatch: {type(s_out)}")
        s_feats, s_logits = s_out[0], s_out[1]

        # --- Teacher forward (HR + depth, no grad) ---
        # data_loader에서 x_lr, x_hr, depth 모두 (INPUT_RESOLUTION)으로 리사이즈되어
        # 공간 정렬이 이미 보장된다. depth 해상도 재정렬은 불필요.
        with torch.no_grad():
            self.teacher.eval()
            if self._teacher_supports_depth:
                d_in = (
                    depth if depth is not None
                    else self._zero_depth_like(x_teacher_rgb)
                )
                t_out = self.teacher(x_teacher_rgb, depth=d_in, is_feat=True)
            else:
                t_out = self.teacher(x_teacher_rgb, is_feat=True)
            t_feats = t_out[0]

        # --- Task loss (CE) ---
        ce_loss = self.ce(s_logits, masks) * self.w_ce_student

        # --- SGSC KD loss (stage-wise) ---
        kd_sum = torch.tensor(0.0, device=device)
        w_sum = 0.0
        stage_loss_dict = {}

        for stage_idx in self.apply_stages:
            f_s_raw = s_feats[stage_idx]
            f_t_raw = t_feats[stage_idx]

            # Channel alignment
            f_s = self.projectors[stage_idx](f_s_raw)

            # Spatial alignment (teacher HR+depth feature → student LR feature 해상도)
            f_t = _interpolate_antialias(f_t_raw, size_hw=f_s.shape[-2:])

            # Stage weight
            w_stage = float(self.stage_weights.get(stage_idx, 1.0))

            # SGSC loss (basis computed from teacher HR+depth feature)
            loss_stage = self.sgsc_loss(f_s=f_s, f_t=f_t)

            kd_sum = kd_sum + w_stage * loss_stage
            w_sum += w_stage
            stage_loss_dict[f"sgsc_s{stage_idx}"] = loss_stage.detach()

        if w_sum > 0:
            kd_loss = (kd_sum / w_sum) * self.w_kd
        else:
            kd_loss = torch.tensor(0.0, device=device)

        total_loss = ce_loss + kd_loss

        out = {
            "total": total_loss,
            "ce_student": ce_loss.detach(),
            "sgsc_kd": kd_loss.detach(),
            "s_logits": s_logits.detach(),
        }
        out.update(stage_loss_dict)

        return out
