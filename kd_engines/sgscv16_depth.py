"""
SGSCv16-depth: plain decoder SGSC baseline for depth-aware teachers.

This keeps the original sgscv16 decoder KD math intact:
  - student LR dec_fuse -> 1x1 channel projector
  - teacher dec_fuse covariance -> top-k PCA basis
  - spatial centering + RMS-normalized coordinate MSE

The only functional difference from sgscv16.py is teacher forwarding:
  teacher(x_teacher, depth=depth, is_feat=True)

Use this as a clean baseline against privileged_hierarchical_semantic_sgsc.py.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from .base_engine import BaseKDEngine


def _autocast_off(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=False)

    class _Dummy:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    return _Dummy()


def _interpolate_antialias(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size_hw:
        return x
    return F.interpolate(
        x, size=size_hw, mode="bilinear", align_corners=False, antialias=True
    )


class SpectralGuidedSpatialPCC(nn.Module):
    def __init__(self, k: int = 64, eps: float = 1e-6):
        super().__init__()
        self.k = int(k)
        self.eps = float(eps)

    def _estimate_basis(self, t_centered: torch.Tensor) -> torch.Tensor:
        with _autocast_off(t_centered.device):
            t_c = t_centered.float()
            b, c, n = t_c.shape
            k = min(self.k, c)
            cov = torch.einsum("bcn,bdn->cd", t_c, t_c) / (float(b * n) + self.eps)
            _, evecs = torch.linalg.eigh(cov)
            return evecs[:, -k:]

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        if f_s.shape != f_t.shape:
            raise RuntimeError(
                f"SGSC feature shapes must match after alignment, got "
                f"{tuple(f_s.shape)} vs {tuple(f_t.shape)}"
            )

        b, c, h, w = f_s.shape
        n = h * w
        with _autocast_off(f_s.device):
            s_flat = f_s.float().reshape(b, c, n)
            t_flat = f_t.float().reshape(b, c, n)
            s_centered = s_flat - s_flat.mean(dim=2, keepdim=True)
            t_centered = t_flat - t_flat.mean(dim=2, keepdim=True)

            with torch.no_grad():
                basis = self._estimate_basis(t_centered).detach()

            k = basis.size(1)
            z_s = torch.einsum("ck,bcn->bkn", basis, s_centered)
            z_t = torch.einsum("ck,bcn->bkn", basis, t_centered)

            rms_s = z_s.pow(2).mean(dim=2, keepdim=True).sqrt()
            rms_t = z_t.pow(2).mean(dim=2, keepdim=True).sqrt()
            z_s = z_s / (rms_s + self.eps)
            z_t = z_t / (rms_t + self.eps)
            return F.mse_loss(z_s, z_t)


class SGSCV16DepthEngine(BaseKDEngine):
    """
    Decoder-only SGSC baseline with depth-aware teacher support.

    Expected train batch:
      imgs=(x_lr, x_hr, depth)

    Defaults:
      student input: x_lr
      teacher input: x_hr
      teacher depth: input depth
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student)
        self.w_ce_student = float(kwargs.get("w_ce_student", 1.0))
        self.w_kd = float(kwargs.get("w_kd", kwargs.get("lambda_transfer", 0.5)))
        self.k = int(kwargs.get("k", 64))
        self.teacher_dec_ch = int(kwargs.get("teacher_dec_ch", 768))
        self.student_dec_ch = int(kwargs.get("student_dec_ch", 256))
        self.teacher_view = str(kwargs.get("teacher_view", "hr")).lower()
        self.teacher_depth_mode = str(kwargs.get("teacher_depth_mode", "input")).lower()
        self.ignore_index = int(
            kwargs.get("ignore_index", getattr(config.DATA, "IGNORE_INDEX", 255))
        )

        if self.teacher_view not in {"hr", "lr"}:
            raise ValueError(f"teacher_view must be 'hr' or 'lr', got {self.teacher_view!r}")
        if self.teacher_depth_mode not in {"input", "zero"}:
            raise ValueError(
                f"teacher_depth_mode must be 'input' or 'zero', got {self.teacher_depth_mode!r}"
            )

        if self.student_dec_ch != self.teacher_dec_ch:
            self.dec_projector = nn.Conv2d(
                self.student_dec_ch, self.teacher_dec_ch, kernel_size=1, bias=True
            )
        else:
            self.dec_projector = nn.Identity()

        self.sgsc_loss = SpectralGuidedSpatialPCC(k=self.k)
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        if hasattr(self.teacher, "set_force_patch_embeds"):
            self.teacher.set_force_patch_embeds(True)
        if hasattr(self.student, "set_force_patch_embeds"):
            self.student.set_force_patch_embeds(True)

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        return self.dec_projector.parameters()

    @staticmethod
    def _unpack_feats_logits_decfuse(out: Any, name: str):
        if not isinstance(out, (tuple, list)) or len(out) < 4:
            raise RuntimeError(
                f"{name} forward(..., is_feat=True) must return "
                "(feats, logits, embeds, dec_fuse)."
            )
        logits = out[1]
        dec = out[3]
        if not torch.is_tensor(logits) or logits.dim() != 4:
            raise RuntimeError(
                f"{name} logits must be 4D tensor, got {getattr(logits, 'shape', None)}"
            )
        if not torch.is_tensor(dec) or dec.dim() != 4:
            raise RuntimeError(
                f"{name} dec_fuse must be 4D tensor, got {getattr(dec, 'shape', None)}"
            )
        return logits, dec

    @staticmethod
    def _unpack_inputs(imgs: Any):
        if not isinstance(imgs, (tuple, list)):
            return imgs, imgs, None
        if len(imgs) >= 3:
            return imgs[0], imgs[1], imgs[2]
        if len(imgs) == 2:
            return imgs[0], imgs[1], None
        if len(imgs) == 1:
            return imgs[0], imgs[0], None
        raise RuntimeError("Empty imgs tuple/list received.")

    @staticmethod
    def _prepare_depth(depth: torch.Tensor | None, x_ref: torch.Tensor) -> torch.Tensor:
        if depth is None:
            return x_ref.new_zeros(x_ref.size(0), 1, x_ref.size(2), x_ref.size(3))
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() != 4:
            raise RuntimeError(f"depth must be (B,H,W) or (B,1,H,W), got {tuple(depth.shape)}")
        if depth.size(1) != 1:
            depth = depth[:, :1]
        depth = depth.to(device=x_ref.device, dtype=x_ref.dtype)
        if depth.shape[-2:] != x_ref.shape[-2:]:
            depth = F.interpolate(depth, size=x_ref.shape[-2:], mode="bilinear", align_corners=False)
        return depth

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        x_lr, x_hr, depth = self._unpack_inputs(imgs)
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        if masks.dim() != 3:
            raise RuntimeError(f"masks must be (B,H,W), got {tuple(masks.shape)}")

        s_out = self.student(x_lr, is_feat=True)
        s_logits, s_dec = self._unpack_feats_logits_decfuse(s_out, "student")

        x_t = x_hr if self.teacher_view == "hr" else x_lr
        depth_t = self._prepare_depth(depth, x_t)
        if self.teacher_depth_mode == "zero":
            depth_t = torch.zeros_like(depth_t)

        with torch.no_grad():
            self.teacher.eval()
            t_out = self.teacher(x_t, depth=depth_t, is_feat=True)
            _t_logits, t_dec = self._unpack_feats_logits_decfuse(t_out, "teacher")

        if s_dec.size(1) != self.student_dec_ch:
            raise RuntimeError(
                f"Student dec_fuse channel mismatch: expected {self.student_dec_ch}, "
                f"got {s_dec.size(1)}"
            )
        if t_dec.size(1) != self.teacher_dec_ch:
            raise RuntimeError(
                f"Teacher dec_fuse channel mismatch: expected {self.teacher_dec_ch}, "
                f"got {t_dec.size(1)}"
            )

        f_s = self.dec_projector(s_dec)
        f_t = _interpolate_antialias(t_dec, size_hw=f_s.shape[-2:]).detach()

        ce = self.ce(s_logits, masks) * self.w_ce_student
        kd_raw = self.sgsc_loss(f_s=f_s, f_t=f_t)
        kd = kd_raw * self.w_kd
        total = ce + kd
        student_miou, student_pa = self._seg_metrics(s_logits, masks)

        return {
            "total": total,
            "ce_student": ce.detach(),
            "sgsc_dec": kd.detach(),
            "sgsc_dec_raw": kd_raw.detach(),
            "student_mIoU": student_miou.detach(),
            "student_pixel_acc": student_pa.detach(),
            "teacher_dec_mean_abs": f_t.detach().abs().mean(),
            "teacher_dec_std": f_t.detach().std(unbiased=False),
            "student_dec_mean_abs": f_s.detach().abs().mean(),
            "student_dec_std": f_s.detach().std(unbiased=False),
            "s_logits": s_logits.detach(),
            "student_input": x_lr.detach(),
            "teacher_input": x_t.detach(),
        }

    def _seg_metrics(
        self, logits: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            num_classes = int(getattr(config.DATA, "NUM_CLASSES", logits.size(1)))
            pred = logits.argmax(dim=1)
            valid = masks != self.ignore_index
            if not valid.any():
                z = logits.new_tensor(0.0)
                return z, z
            pred_v = pred[valid].clamp(0, num_classes - 1)
            mask_v = masks[valid].clamp(0, num_classes - 1)
            pixel_acc = (pred_v == mask_v).float().mean()
            idx = mask_v * num_classes + pred_v
            conf = torch.bincount(idx, minlength=num_classes * num_classes)
            conf = conf.reshape(num_classes, num_classes).float()
            inter = conf.diag()
            union = conf.sum(dim=1) + conf.sum(dim=0) - inter
            valid_cls = union > 0
            miou = torch.where(valid_cls, inter / union.clamp_min(1.0), torch.zeros_like(union))
            miou = miou[valid_cls].mean() if valid_cls.any() else logits.new_tensor(0.0)
            return miou, pixel_acc
