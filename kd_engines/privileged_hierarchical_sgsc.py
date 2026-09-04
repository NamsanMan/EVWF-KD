from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

import config
from .privileged_transfer_sgsc import (
    PrivilegedTransferSGSCEngine,
    _interpolate_antialias,
)
from .sgscv5r2_depth import SpectralGuidedSpatialPCC, _safe_groupnorm


def _parse_stages(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(x) for x in value)


class PrivilegedHierarchicalSGSCEngine(PrivilegedTransferSGSCEngine):
    """
    Hierarchical privileged SGSC KD.

    This combines two existing code paths while keeping their roles intact:

    1. dec_fuse transferable-subspace KD from PrivilegedTransferSGSCEngine:
       teacher HR+depth -> dec_fuse_t -> frozen transfer_projector -> G_t
       student LR       -> dec_fuse_s -> trainable projector         -> G_s

    2. stage-wise full-feature SGSC from SGSCDepthEngine:
       teacher HR+depth encoder features provide a high-level spectral basis;
       student LR encoder features are channel-aligned and projected onto it.

    The teacher is frozen and strictly receives x_hr + depth. The student
    strictly receives x_lr only.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student, **kwargs)

        self.lambda_transfer = float(
            kwargs.get("lambda_transfer", kwargs.get("lambda_kd", 0.5))
        )
        self.lambda_stage = float(kwargs.get("lambda_stage", kwargs.get("w_stage_kd", 0.25)))

        self.k_stage = int(kwargs.get("k_stage", kwargs.get("k", 64)))
        self.apply_stages = _parse_stages(kwargs.get("apply_stages", (3,)))
        self.stage_weights = kwargs.get("stage_weights", {3: 1.0})

        student_ch = kwargs.get("student_channels", [32, 64, 160, 256])
        teacher_ch = kwargs.get("teacher_channels", [64, 128, 320, 512])

        self.stage_projectors = nn.ModuleList()
        for i in range(4):
            if i in self.apply_stages:
                out_ch = int(teacher_ch[i])
                self.stage_projectors.append(
                    nn.Sequential(
                        nn.Conv2d(int(student_ch[i]), out_ch, kernel_size=1, bias=True),
                        _safe_groupnorm(out_ch, max_groups=32),
                    )
                )
            else:
                self.stage_projectors.append(nn.Identity())

        self.stage_sgsc_loss = SpectralGuidedSpatialPCC(k=self.k_stage)

        if hasattr(self.teacher, "set_force_patch_embeds"):
            self.teacher.set_force_patch_embeds(True)
        if hasattr(self.student, "set_force_patch_embeds"):
            self.student.set_force_patch_embeds(True)

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.student_projector is not None:
            params.extend(self.student_projector.parameters())
        params.extend(self.stage_projectors.parameters())
        return params

    @staticmethod
    def _extract_feats_logits_decfuse(model_out: Any, name: str):
        if not isinstance(model_out, (tuple, list)) or len(model_out) < 4:
            raise RuntimeError(
                f"{name} forward(..., is_feat=True) must return "
                "(fused_feats, logits, embeds, dec_fuse)."
            )
        feats, logits, dec_fuse = model_out[0], model_out[1], model_out[3]
        if not isinstance(feats, (tuple, list)):
            raise RuntimeError(f"{name} output[0] must be a stage feature list/tuple.")
        if not torch.is_tensor(logits):
            raise RuntimeError(f"{name} output[1] must be logits tensor.")
        if not torch.is_tensor(dec_fuse):
            raise RuntimeError(f"{name} output[3] must be dec_fuse tensor.")
        return feats, logits, dec_fuse

    def _stage_weight(self, stage_idx: int) -> float:
        if isinstance(self.stage_weights, dict):
            return float(
                self.stage_weights.get(stage_idx, self.stage_weights.get(str(stage_idx), 1.0))
            )
        return 1.0

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        x_lr, x_hr, depth = self._unpack_inputs(imgs)
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        if masks.dim() != 3:
            raise RuntimeError(f"masks must be (B,H,W), got {tuple(masks.shape)}")

        s_out = self.student(x_lr, is_feat=True)
        s_feats, s_logits, dec_fuse_s = self._extract_feats_logits_decfuse(s_out, "student")

        with torch.no_grad():
            self.teacher.eval()
            t_out = self.teacher(x_hr, depth=depth, is_feat=True)
            t_feats, _t_logits_unused, dec_fuse_t = self._extract_feats_logits_decfuse(
                t_out, "teacher"
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
        transfer_raw = self.sgsc_loss(g_s, g_t)
        transfer_kd = transfer_raw * self.lambda_transfer

        stage_sum = torch.tensor(0.0, device=device)
        stage_weight_sum = 0.0
        stage_loss_dict: Dict[str, torch.Tensor] = {}

        for stage_idx in self.apply_stages:
            if stage_idx < 0 or stage_idx >= len(s_feats) or stage_idx >= len(t_feats):
                raise RuntimeError(
                    f"Invalid stage index {stage_idx}; "
                    f"student stages={len(s_feats)}, teacher stages={len(t_feats)}"
                )
            f_s_raw = s_feats[stage_idx]
            f_t_raw = t_feats[stage_idx]
            f_s = self.stage_projectors[stage_idx](f_s_raw)
            f_t = _interpolate_antialias(f_t_raw, size_hw=f_s.shape[-2:]).detach()

            loss_stage = self.stage_sgsc_loss(f_s=f_s, f_t=f_t)
            w_stage = self._stage_weight(stage_idx)
            stage_sum = stage_sum + w_stage * loss_stage
            stage_weight_sum += w_stage
            stage_loss_dict[f"hier_sgsc_s{stage_idx}"] = loss_stage.detach()

        if stage_weight_sum > 0:
            stage_raw = stage_sum / stage_weight_sum
        else:
            stage_raw = torch.tensor(0.0, device=device)
        stage_kd = stage_raw * self.lambda_stage

        total = ce + transfer_kd + stage_kd
        student_miou, student_pa = self._seg_metrics(s_logits, masks)

        out: Dict[str, Any] = {
            "total": total,
            "ce_student": ce.detach(),
            "transfer_sgsc": transfer_kd.detach(),
            "transfer_sgsc_raw": transfer_raw.detach(),
            "hier_stage_sgsc": stage_kd.detach(),
            "hier_stage_sgsc_raw": stage_raw.detach(),
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
        out.update(stage_loss_dict)
        return out
