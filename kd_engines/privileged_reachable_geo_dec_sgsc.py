from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .privileged_hierarchical_sgsc import PrivilegedHierarchicalSGSCEngine
from .privileged_reachable_geo_sgsc import ReachableGeometryStageSGSCLoss
from .privileged_transfer_sgsc import _interpolate_antialias
from .sgscv16_depth import SpectralGuidedSpatialPCC as DecoderSGSCLoss


class PrivilegedReachableGeoDecSGSCEngine(PrivilegedHierarchicalSGSCEngine):
    """
    Final combined engine:

    Encoder:
        LR-reachable privileged geometry SGSC.

    Decoder:
        sgscv16_depth-style dec_fuse SGSC against the depth-aware teacher.

    It reuses the encoder DS-CSF implementation from privileged_reachable_geo_sgsc.py
    and the decoder-only sgscv16_depth loss.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student, **kwargs)

        self.lambda_stage = float(kwargs.get("lambda_stage", kwargs.get("w_stage_kd", 0.5)))
        self.lambda_decoder = float(
            kwargs.get("lambda_decoder", kwargs.get("w_dec_kd", kwargs.get("w_kd", 0.5)))
        )

        self.k_decoder = int(kwargs.get("k_decoder", 64))
        self.teacher_dec_ch = int(kwargs.get("teacher_dec_ch", 768))
        self.student_dec_ch = int(kwargs.get("student_dec_ch", 256))
        self.decoder_teacher_view = str(kwargs.get("decoder_teacher_view", "hr")).lower()
        self.decoder_depth_mode = str(kwargs.get("decoder_depth_mode", "input")).lower()

        if self.decoder_teacher_view not in {"hr", "lr"}:
            raise ValueError(
                "decoder_teacher_view must be 'hr' or 'lr', "
                f"got {self.decoder_teacher_view!r}"
            )
        if self.decoder_depth_mode not in {"input", "zero"}:
            raise ValueError(
                "decoder_depth_mode must be 'input' or 'zero', "
                f"got {self.decoder_depth_mode!r}"
            )

        if self.student_dec_ch != self.teacher_dec_ch:
            self.dec_projector = nn.Conv2d(
                self.student_dec_ch, self.teacher_dec_ch, kernel_size=1, bias=True
            )
        else:
            self.dec_projector = nn.Identity()

        self.decoder_sgsc_loss = DecoderSGSCLoss(k=self.k_decoder)
        teacher_ch = kwargs.get("teacher_channels", [64, 128, 320, 512])
        self.stage_sgsc_loss = ReachableGeometryStageSGSCLoss(
            k=self.k_stage,
            stage_channels=teacher_ch,
            csf_hidden_ch=int(kwargs.get("rg_csf_hidden_ch", 64)),
            csf_reduction=int(kwargs.get("rg_csf_reduction", 4)),
            lr_scale_alpha=float(kwargs.get("rg_lr_scale_alpha", 1.0)),
            depth_gate_alpha=float(kwargs.get("rg_depth_gate_alpha", 1.0)),
            depth_similarity_power=float(kwargs.get("rg_depth_similarity_power", 1.0)),
            weight_min=float(kwargs.get("rg_weight_min", 0.5)),
            weight_max=float(kwargs.get("rg_weight_max", 2.0)),
            eps=float(kwargs.get("eps", 1e-6)),
        )

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(self.stage_projectors.parameters())
        params.extend(self.dec_projector.parameters())
        params.extend(self.stage_sgsc_loss.parameters())
        return params

    @staticmethod
    def _zero_depth_like(x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.size(0), 1, x.size(2), x.size(3))

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _teacher_supports_lr_condition(self) -> bool:
        inner = self._unwrap_model(self.teacher)
        return bool(getattr(inner, "supports_lr_condition", False))

    def _teacher_hr_depth_forward(
        self,
        x_hr: torch.Tensor,
        depth: torch.Tensor,
        x_lr: torch.Tensor,
    ):
        if self._teacher_supports_lr_condition():
            return self.teacher(x_hr, depth=depth, lr_condition=x_lr, is_feat=True)
        return self.teacher(x_hr, depth=depth, is_feat=True)

    def _teacher_forward_with_optional_lr_condition(
        self,
        x: torch.Tensor,
        depth: torch.Tensor,
        lr_condition: torch.Tensor | None = None,
    ):
        if lr_condition is not None and self._teacher_supports_lr_condition():
            return self.teacher(x, depth=depth, lr_condition=lr_condition, is_feat=True)
        return self.teacher(x, depth=depth, is_feat=True)

    def _teacher_lr_rgb_only_forward(self, x_lr: torch.Tensor):
        if all(hasattr(self.teacher, name) for name in ("_encode_rgb", "model")):
            for i in range(4):
                if hasattr(self.teacher, "_patch_embed_cache"):
                    self.teacher._patch_embed_cache[i] = None
                if hasattr(self.teacher, "_patch_hw_cache"):
                    self.teacher._patch_hw_cache[i] = None
            if hasattr(self.teacher, "_last_decoder_fuse"):
                self.teacher._last_decoder_fuse = None
            if hasattr(self.teacher, "_last_fusion_stats"):
                self.teacher._last_fusion_stats = {}
            rgb_feats, embeds = self.teacher._encode_rgb(x_lr)
            logits_low = self.teacher.model.decode_head(rgb_feats)
            logits = F.interpolate(logits_low, size=x_lr.shape[-2:], mode="bilinear", align_corners=False)
            dec_fuse = getattr(self.teacher, "_last_decoder_fuse", None)
            return rgb_feats, logits, embeds, dec_fuse

        zero_depth = self._zero_depth_like(x_lr)
        return self.teacher(x_lr, depth=zero_depth, is_feat=True)

    def _decoder_teacher_forward(
        self,
        x_lr: torch.Tensor,
        x_hr: torch.Tensor,
        depth: torch.Tensor,
        t_hd_dec: torch.Tensor,
        t_lz_dec: torch.Tensor,
    ) -> torch.Tensor:
        if self.decoder_teacher_view == "hr" and self.decoder_depth_mode == "input":
            return t_hd_dec
        if self.decoder_teacher_view == "lr" and self.decoder_depth_mode == "zero":
            return t_lz_dec

        x_t = x_hr if self.decoder_teacher_view == "hr" else x_lr
        if self.decoder_depth_mode == "zero":
            d_t = self._zero_depth_like(x_t)
        else:
            d_t = depth

        with torch.no_grad():
            self.teacher.eval()
            lr_condition = (
                x_lr
                if self.decoder_teacher_view == "hr" and self.decoder_depth_mode == "input"
                else None
            )
            out = self._teacher_forward_with_optional_lr_condition(
                x_t, depth=d_t, lr_condition=lr_condition
            )
            _feats, _logits, dec = self._extract_feats_logits_decfuse(
                out, "teacher_decoder_target"
            )
        return dec

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        x_lr, x_hr, depth = self._unpack_inputs(imgs)
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        if masks.dim() != 3:
            raise RuntimeError(f"masks must be (B,H,W), got {tuple(masks.shape)}")

        s_out = self.student(x_lr, is_feat=True)
        s_feats, s_logits, dec_s = self._extract_feats_logits_decfuse(s_out, "student")

        with torch.no_grad():
            self.teacher.eval()
            t_hd_out = self._teacher_hr_depth_forward(x_hr, depth=depth, x_lr=x_lr)
            t_hd_feats, _t_hd_logits, dec_hd = self._extract_feats_logits_decfuse(
                t_hd_out, "teacher_hr_depth"
            )
            hr_teacher_stats = {}
            inner_teacher = self._unwrap_model(self.teacher)
            if self._teacher_supports_lr_condition() and hasattr(
                inner_teacher, "get_last_fusion_stats"
            ):
                stats = inner_teacher.get_last_fusion_stats()
                if isinstance(stats, dict):
                    hr_teacher_stats = {
                        f"teacher_hr_{key}": value.detach() if torch.is_tensor(value) else value
                        for key, value in stats.items()
                    }
            t_lz_out = self._teacher_lr_rgb_only_forward(x_lr)
            t_lz_feats, _t_lz_logits, dec_lz = self._extract_feats_logits_decfuse(
                t_lz_out, "teacher_lr_rgb_only"
            )

        ce = self.ce(s_logits, masks) * self.w_ce_student

        if dec_s.size(1) != self.student_dec_ch:
            raise RuntimeError(
                f"Student dec_fuse channel mismatch: expected {self.student_dec_ch}, "
                f"got {dec_s.size(1)}"
            )
        dec_t = self._decoder_teacher_forward(x_lr, x_hr, depth, dec_hd, dec_lz).detach()
        if dec_t.size(1) != self.teacher_dec_ch:
            raise RuntimeError(
                f"Teacher dec_fuse channel mismatch: expected {self.teacher_dec_ch}, "
                f"got {dec_t.size(1)}"
            )

        dec_s_proj = self.dec_projector(dec_s)
        dec_t = _interpolate_antialias(dec_t, size_hw=dec_s_proj.shape[-2:]).detach()
        dec_raw = self.decoder_sgsc_loss(f_s=dec_s_proj, f_t=dec_t)
        dec_kd = dec_raw * self.lambda_decoder

        stage_sum = torch.tensor(0.0, device=device)
        stage_weight_sum = 0.0
        stage_loss_dict: Dict[str, torch.Tensor] = {}
        weight_means = []
        weight_stds = []
        weight_mins = []
        weight_maxes = []
        weight_low_clamp_fracs = []
        weight_high_clamp_fracs = []
        weighted_unweighted_ratios = []
        unweighted_mses = []
        geometry_stds = []
        reach_stds = []
        reach_mins = []
        hd_lz_mses = []
        hd_lz_abs_means = []

        for stage_idx in self.apply_stages:
            if (
                stage_idx < 0
                or stage_idx >= len(s_feats)
                or stage_idx >= len(t_hd_feats)
                or stage_idx >= len(t_lz_feats)
            ):
                raise RuntimeError(
                    f"Invalid stage index {stage_idx}; student={len(s_feats)}, "
                    f"teacher_hd={len(t_hd_feats)}, teacher_lz={len(t_lz_feats)}"
                )

            f_s = self.stage_projectors[stage_idx](s_feats[stage_idx])
            f_hd = _interpolate_antialias(t_hd_feats[stage_idx], size_hw=f_s.shape[-2:]).detach()
            f_lr_rgb = _interpolate_antialias(
                t_lz_feats[stage_idx], size_hw=f_s.shape[-2:]
            ).detach()

            loss_stage, diag = self.stage_sgsc_loss(
                f_s=f_s,
                f_hd=f_hd,
                f_lr_rgb=f_lr_rgb,
                depth=depth,
                stage_idx=stage_idx,
            )
            w_stage = self._stage_weight(stage_idx)
            stage_sum = stage_sum + w_stage * loss_stage
            stage_weight_sum += w_stage

            prefix = f"reachable_geo_s{stage_idx}"
            stage_loss_dict[prefix] = loss_stage.detach()
            for name, value in diag.items():
                stage_loss_dict[f"{prefix}_{name}"] = value.detach()

            weight_means.append(diag["weight_mean"].detach())
            weight_stds.append(diag["weight_std"].detach())
            weight_mins.append(diag["weight_min"].detach())
            weight_maxes.append(diag["weight_max"].detach())
            weight_low_clamp_fracs.append(diag["weight_low_clamp_frac"].detach())
            weight_high_clamp_fracs.append(diag["weight_high_clamp_frac"].detach())
            weighted_unweighted_ratios.append(diag["weighted_unweighted_ratio"].detach())
            unweighted_mses.append(diag["unweighted_mse"].detach())
            geometry_stds.append(diag["geometry_std"].detach())
            reach_stds.append(diag["reach_std"].detach())
            reach_mins.append(diag["reach_min"].detach())
            hd_lz_mses.append(diag["hd_lz_coord_mse"].detach())
            hd_lz_abs_means.append(diag["hd_lz_coord_abs_mean"].detach())

        if stage_weight_sum > 0:
            stage_raw = stage_sum / stage_weight_sum
        else:
            stage_raw = torch.tensor(0.0, device=device)
        stage_kd = stage_raw * self.lambda_stage

        total = ce + dec_kd + stage_kd
        student_miou, student_pa = self._seg_metrics(s_logits, masks)

        out: Dict[str, Any] = {
            "total": total,
            "ce_student": ce.detach(),
            "sgsc_dec": dec_kd.detach(),
            "sgsc_dec_raw": dec_raw.detach(),
            "reachable_geo_stage_sgsc": stage_kd.detach(),
            "reachable_geo_stage_sgsc_raw": stage_raw.detach(),
            "student_mIoU": student_miou.detach(),
            "student_pixel_acc": student_pa.detach(),
            "teacher_dec_mean_abs": dec_t.detach().abs().mean(),
            "teacher_dec_std": dec_t.detach().std(unbiased=False),
            "student_dec_mean_abs": dec_s_proj.detach().abs().mean(),
            "student_dec_std": dec_s_proj.detach().std(unbiased=False),
            "s_logits": s_logits.detach(),
            "student_input": x_lr.detach(),
            "teacher_input": x_hr.detach(),
        }
        out.update(hr_teacher_stats)
        if weight_means:
            out["rg_weight_mean"] = torch.stack(weight_means).mean()
            out["rg_weight_std"] = torch.stack(weight_stds).mean()
            out["rg_weight_min"] = torch.stack(weight_mins).min()
            out["rg_weight_max"] = torch.stack(weight_maxes).max()
            out["rg_weight_low_clamp_frac"] = torch.stack(weight_low_clamp_fracs).mean()
            out["rg_weight_high_clamp_frac"] = torch.stack(weight_high_clamp_fracs).mean()
            out["rg_weighted_unweighted_ratio"] = torch.stack(weighted_unweighted_ratios).mean()
            out["rg_unweighted_mse"] = torch.stack(unweighted_mses).mean()
            out["rg_geometry_std"] = torch.stack(geometry_stds).mean()
            out["rg_reach_std"] = torch.stack(reach_stds).mean()
            out["rg_reach_min"] = torch.stack(reach_mins).min()
            out["rg_hd_lz_coord_mse"] = torch.stack(hd_lz_mses).mean()
            out["rg_hd_lz_coord_abs_mean"] = torch.stack(hd_lz_abs_means).mean()
        out.update(stage_loss_dict)
        return out
