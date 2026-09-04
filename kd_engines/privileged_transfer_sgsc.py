from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import inspect
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


def _gn_groups(channels: int, max_groups: int = 8) -> int:
    groups = max(1, min(int(max_groups), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _interpolate_antialias(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size_hw:
        return x
    try:
        return F.interpolate(
            x, size=size_hw, mode="bilinear", align_corners=False, antialias=True
        )
    except TypeError:
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _supports_depth_forward(model: nn.Module) -> bool:
    m = _unwrap_model(model)
    try:
        return "depth" in inspect.signature(m.forward).parameters
    except Exception:
        return False


class _ProjectionHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch, groups), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransferableSGSCLoss(nn.Module):
    """
    SGSC-style spatial PCC loss whose PCA basis is estimated only from
    teacher transferable feature G_t.
    """

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
            _, eigenvectors = torch.linalg.eigh(cov)
            return eigenvectors[:, -k:]

    def forward(self, g_s: torch.Tensor, g_t: torch.Tensor) -> torch.Tensor:
        if g_s.shape != g_t.shape:
            raise RuntimeError(f"SGSC feature shapes must match, got {tuple(g_s.shape)} vs {tuple(g_t.shape)}")

        b, c, h, w = g_s.shape
        n = h * w
        with _autocast_off(g_s.device):
            s_flat = g_s.float().reshape(b, c, n)
            t_flat = g_t.float().reshape(b, c, n)
            s_centered = s_flat - s_flat.mean(dim=2, keepdim=True)
            t_centered = t_flat - t_flat.mean(dim=2, keepdim=True)

            with torch.no_grad():
                basis = self._estimate_basis(t_centered).detach()

            k = int(basis.size(1))
            z_s = torch.einsum("ck,bcn->bkn", basis, s_centered)
            z_t = torch.einsum("ck,bcn->bkn", basis, t_centered)

            rms_s = z_s.pow(2).mean(dim=2, keepdim=True).sqrt()
            rms_t = z_t.pow(2).mean(dim=2, keepdim=True).sqrt()
            z_s = z_s / (rms_s + self.eps)
            z_t = z_t / (rms_t + self.eps)
            return F.mse_loss(z_s, z_t)


class PrivilegedTransferSGSCEngine(BaseKDEngine):
    """
    KD from a frozen privileged HR+depth teacher's learned transferable
    geometry-semantic branch G only.

    Teacher:
        teacher(x_hr, depth=depth, is_feat=True) -> dec_fuse_t
        frozen teacher_transfer_projector(dec_fuse_t) -> G_t

    Student:
        student(x_lr, is_feat=True) -> dec_fuse_s
        trainable student_projector(dec_fuse_s) -> G_s

    Loss:
        CE(student_logits, y) + lambda_kd * SGSC(G_s, G_t)
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student)
        self.w_ce_student = float(kwargs.get("w_ce_student", 1.0))
        self.lambda_kd = float(kwargs.get("lambda_kd", kwargs.get("w_kd", 0.5)))
        self.k = int(kwargs.get("k", 64))
        self.ignore_index = int(kwargs.get("ignore_index", getattr(config.DATA, "IGNORE_INDEX", 255)))
        self.norm_groups = int(kwargs.get("norm_groups", 8))

        ckpt_arg = kwargs.get("teacher_projector_ckpt", "")
        self.teacher_projector_ckpt = str(ckpt_arg).strip() if ckpt_arg is not None else ""
        if not self.teacher_projector_ckpt:
            self.teacher_projector_ckpt = str(kwargs.get("teacher_ckpt", getattr(config, "TEACHER_CKPT", "")))
        self.teacher_projector_prefix = str(kwargs.get("teacher_projector_prefix", "transfer_projector"))

        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.sgsc_loss = TransferableSGSCLoss(k=self.k)

        self.teacher_transfer_projector: Optional[nn.Module] = None
        self.student_projector: Optional[nn.Module] = None
        self._teacher_projector_loaded = False

        self._teacher_supports_depth = _supports_depth_forward(self.teacher)
        if not self._teacher_supports_depth:
            raise RuntimeError("PrivilegedTransferSGSCEngine requires a teacher forward with depth=...")

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        if self.student_projector is None:
            return []
        return self.student_projector.parameters()

    @staticmethod
    def _unpack_inputs(imgs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(imgs, (tuple, list)) or len(imgs) < 3:
            raise RuntimeError("PrivilegedTransferSGSCEngine requires imgs=(x_lr, x_hr, depth).")
        return imgs[0], imgs[1], imgs[2]

    @staticmethod
    def _extract_logits_decfuse(model_out: Any, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(model_out, (tuple, list)) or len(model_out) < 4:
            raise RuntimeError(f"{name} forward(..., is_feat=True) must return (feats, logits, embeds, dec_fuse).")
        logits = model_out[1]
        dec_fuse = model_out[3]
        if not torch.is_tensor(logits):
            raise RuntimeError(f"{name} output[1] must be logits tensor.")
        if not torch.is_tensor(dec_fuse):
            raise RuntimeError(f"{name} output[3] must be dec_fuse tensor.")
        return logits, dec_fuse

    @staticmethod
    def _as_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            for key in ("transfer_projector", "transfer_projector_state", "teacher_transfer_projector"):
                if key in obj and isinstance(obj[key], dict):
                    return obj[key]
            for key in ("kd_engine_state", "engine_state", "state_dict"):
                if key in obj and isinstance(obj[key], dict):
                    nested = obj[key]
                    matches = {k: v for k, v in nested.items() if "transfer_projector" in k}
                    if matches:
                        return matches
            return {k: v for k, v in obj.items() if torch.is_tensor(v)}
        raise RuntimeError(f"Unsupported checkpoint object: {type(obj)}")

    def _strip_projector_prefix(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        candidates = []
        prefix = self.teacher_projector_prefix
        for k in state:
            if k.startswith(prefix + "."):
                candidates.append((prefix + ".", k[len(prefix) + 1 :]))
            if k.startswith("teacher_transfer_projector."):
                candidates.append(("teacher_transfer_projector.", k[len("teacher_transfer_projector.") :]))
            if ".transfer_projector." in k:
                suffix = k.split(".transfer_projector.", 1)[1]
                candidates.append((k[: -len(suffix)], suffix))
        if candidates:
            stripped = {}
            for original in state:
                new_key = original
                if original.startswith(prefix + "."):
                    new_key = original[len(prefix) + 1 :]
                elif original.startswith("teacher_transfer_projector."):
                    new_key = original[len("teacher_transfer_projector.") :]
                elif ".transfer_projector." in original:
                    new_key = original.split(".transfer_projector.", 1)[1]
                stripped[new_key] = state[original]
            return stripped
        return dict(state)

    @staticmethod
    def _infer_projector_channels(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        first = state.get("net.0.weight")
        last = state.get("net.3.weight")
        if first is None or last is None:
            raise RuntimeError(
                "Transfer projector checkpoint must contain net.0.weight and net.3.weight. "
                "This usually means the teacher-training checkpoint did not save transfer_projector."
            )
        in_ch = int(first.shape[1])
        out_ch = int(last.shape[0])
        return in_ch, out_ch

    def _load_teacher_projector(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._teacher_projector_loaded:
            return
        if not self.teacher_projector_ckpt:
            raise RuntimeError("teacher_projector_ckpt is empty; trained transfer_projector weights are required.")

        path = Path(self.teacher_projector_ckpt)
        if not path.exists():
            raise FileNotFoundError(f"teacher_projector_ckpt not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        raw_state = self._as_state_dict(ckpt)
        transfer_like = {k: v for k, v in raw_state.items() if "transfer_projector" in k}
        state = self._strip_projector_prefix(transfer_like if transfer_like else raw_state)

        in_ch, out_ch = self._infer_projector_channels(state)
        projector = _ProjectionHead(in_ch, out_ch, self.norm_groups)
        missing, unexpected = projector.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Failed to load teacher transfer projector: missing={missing}, unexpected={unexpected}")
        projector.to(device=device, dtype=dtype)
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False

        self.teacher_transfer_projector = projector
        self._teacher_projector_loaded = True

    def _build_student_projector(self, student_dec_fuse: torch.Tensor, out_ch: int) -> None:
        if self.student_projector is not None:
            return
        in_ch = int(student_dec_fuse.size(1))
        self.student_projector = _ProjectionHead(in_ch, int(out_ch), self.norm_groups).to(
            device=student_dec_fuse.device,
            dtype=student_dec_fuse.dtype,
        )

    def _seg_metrics(self, logits: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)
            valid = masks != self.ignore_index
            if not valid.any():
                z = logits.new_tensor(0.0)
                return z, z
            pred_v = pred[valid]
            mask_v = masks[valid]
            pixel_acc = (pred_v == mask_v).float().mean()
            num_classes = int(getattr(config.DATA, "NUM_CLASSES", logits.size(1)))
            pred_v = pred_v.clamp(0, num_classes - 1)
            mask_v = mask_v.clamp(0, num_classes - 1)
            idx = mask_v * num_classes + pred_v
            conf = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes).float()
            inter = conf.diag()
            union = conf.sum(dim=1) + conf.sum(dim=0) - inter
            keep = union > 0
            miou = (inter[keep] / union[keep].clamp_min(1.0)).mean() if keep.any() else logits.new_tensor(0.0)
            return miou, pixel_acc

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        x_lr, x_hr, depth = self._unpack_inputs(imgs)
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        if masks.dim() != 3:
            raise RuntimeError(f"masks must be (B,H,W), got {tuple(masks.shape)}")

        s_logits, dec_fuse_s = self._extract_logits_decfuse(self.student(x_lr, is_feat=True), "student")

        with torch.no_grad():
            self.teacher.eval()
            t_logits_unused, dec_fuse_t = self._extract_logits_decfuse(
                self.teacher(x_hr, depth=depth, is_feat=True), "teacher"
            )
            self._load_teacher_projector(device=dec_fuse_t.device, dtype=dec_fuse_t.dtype)
            assert self.teacher_transfer_projector is not None
            self.teacher_transfer_projector.eval()
            g_t = self.teacher_transfer_projector(dec_fuse_t).detach()

        self._build_student_projector(dec_fuse_s, out_ch=int(g_t.size(1)))
        assert self.student_projector is not None
        g_s = self.student_projector(dec_fuse_s)

        g_t = _interpolate_antialias(g_t, size_hw=g_s.shape[-2:]).detach()
        ce = self.ce(s_logits, masks) * self.w_ce_student
        sgsc_raw = self.sgsc_loss(g_s, g_t)
        sgsc = sgsc_raw * self.lambda_kd
        total = ce + sgsc

        student_miou, student_pa = self._seg_metrics(s_logits, masks)

        return {
            "total": total,
            "ce_student": ce.detach(),
            "transfer_sgsc": sgsc.detach(),
            "transfer_sgsc_raw": sgsc_raw.detach(),
            "student_mIoU": student_miou.detach(),
            "student_pixel_acc": student_pa.detach(),
            "teacher_G_mean_abs": g_t.detach().abs().mean(),
            "teacher_G_std": g_t.detach().std(unbiased=False),
            "student_G_mean_abs": g_s.detach().abs().mean(),
            "student_G_std": g_s.detach().std(unbiased=False),
            "s_logits": s_logits.detach(),
            "student_input": x_lr.detach(),
            "teacher_input": x_hr.detach(),
        }
