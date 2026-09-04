from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn

from .privileged_reachable_geo_dec_sgsc import PrivilegedReachableGeoDecSGSCEngine
from .privileged_reachable_geo_sgsc import (
    DepthSimilarityCSF,
    ReachableGeometryStageSGSCLoss,
)
from .privileged_transfer_sgsc import _autocast_off


def _parse_stage_k(value, default_k: int, num_stages: int) -> dict[int, int]:
    if value is None:
        return {idx: int(default_k) for idx in range(num_stages)}

    if isinstance(value, dict):
        return {
            int(idx): int(value.get(idx, value.get(str(idx), default_k)))
            for idx in range(num_stages)
        }

    if isinstance(value, (list, tuple)):
        out: dict[int, int] = {}
        for idx in range(num_stages):
            out[idx] = int(value[idx]) if idx < len(value) else int(default_k)
        return out

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {idx: int(default_k) for idx in range(num_stages)}
        out = {idx: int(default_k) for idx in range(num_stages)}
        # Accept both "0:32,1:64,2:96,3:128" and "32,64,96,128".
        if ":" in value:
            for item in value.split(","):
                if not item.strip():
                    continue
                key, val = item.split(":", 1)
                out[int(key.strip())] = int(val.strip())
        else:
            vals = [int(v.strip()) for v in value.split(",") if v.strip()]
            for idx, val in enumerate(vals[:num_stages]):
                out[idx] = val
        return out

    return {idx: int(default_k) for idx in range(num_stages)}


class StagewiseKReachableGeometryStageSGSCLoss(ReachableGeometryStageSGSCLoss):
    """
    Fixed-k reachable geometry encoder KD with a separate k for each stage.

    This keeps the original priv_reach_geo_dec_sgsc behavior, but replaces the
    single encoder k_stage with stage-wise k values. It is useful when earlier
    stages should use a compact basis while deeper stages can keep more
    coordinates.
    """

    def __init__(
        self,
        k_by_stage: dict[int, int] | list[int] | tuple[int, ...] | str | None = None,
        default_k: int = 64,
        stage_channels: list[int] | tuple[int, ...] | None = None,
        csf_hidden_ch: int = 64,
        csf_reduction: int = 4,
        lr_scale_alpha: float = 1.0,
        depth_gate_alpha: float = 1.0,
        depth_similarity_power: float = 1.0,
        weight_min: float = 0.5,
        weight_max: float = 2.0,
        eps: float = 1e-6,
    ):
        stage_channels = list(stage_channels or [64, 128, 320, 512])
        super().__init__(
            k=default_k,
            stage_channels=None,
            csf_hidden_ch=csf_hidden_ch,
            csf_reduction=csf_reduction,
            lr_scale_alpha=lr_scale_alpha,
            depth_gate_alpha=depth_gate_alpha,
            depth_similarity_power=depth_similarity_power,
            weight_min=weight_min,
            weight_max=weight_max,
            eps=eps,
        )
        self.k_by_stage = _parse_stage_k(k_by_stage, default_k, len(stage_channels))
        self.stage_channels = stage_channels
        self.csf_modules = nn.ModuleDict()
        for idx, ch in enumerate(stage_channels):
            k = min(int(self.k_by_stage.get(idx, default_k)), int(ch))
            self.csf_modules[str(idx)] = DepthSimilarityCSF(
                channels=int(ch),
                k=k,
                hidden_ch=csf_hidden_ch,
                reduction=csf_reduction,
                mask_min=weight_min,
                mask_max=weight_max,
                lr_scale_alpha=lr_scale_alpha,
                depth_gate_alpha=depth_gate_alpha,
                depth_similarity_power=depth_similarity_power,
                eps=eps,
            )

    def _stage_k(self, stage_idx: int, channels: int) -> int:
        return max(1, min(int(self.k_by_stage.get(int(stage_idx), self.k)), int(channels)))

    def _estimate_basis(self, t_centered: torch.Tensor, stage_idx: int) -> torch.Tensor:
        with _autocast_off(t_centered.device):
            t_c = t_centered.float()
            b, c, n = t_c.shape
            k = self._stage_k(stage_idx, c)
            cov = torch.einsum("bcn,bdn->cd", t_c, t_c) / (float(b * n) + self.eps)
            _, eigenvectors = torch.linalg.eigh(cov)
            return eigenvectors[:, -k:]

    def forward(
        self,
        f_s: torch.Tensor,
        f_hd: torch.Tensor,
        f_lr_rgb: torch.Tensor,
        depth: torch.Tensor,
        stage_idx: int,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if f_s.shape != f_hd.shape or f_s.shape != f_lr_rgb.shape:
            raise RuntimeError(
                "Stage-wise k reachable geometry SGSC feature shapes must match, got "
                f"student={tuple(f_s.shape)}, hr_depth={tuple(f_hd.shape)}, "
                f"lr_rgb={tuple(f_lr_rgb.shape)}"
            )
        key = str(int(stage_idx))
        if key not in self.csf_modules:
            raise RuntimeError(f"DS-CSF module for stage {stage_idx} is not initialized.")

        b, c, h, w = f_s.shape
        n = h * w
        depth = self._prepare_depth(depth, size_hw=(h, w)).to(device=f_s.device)

        with _autocast_off(f_s.device):
            s_flat = f_s.float().reshape(b, c, n)
            hd_flat = f_hd.float().reshape(b, c, n)
            lr_flat = f_lr_rgb.float().reshape(b, c, n)

            s_centered = s_flat - s_flat.mean(dim=2, keepdim=True)
            hd_centered = hd_flat - hd_flat.mean(dim=2, keepdim=True)
            lr_centered = lr_flat - lr_flat.mean(dim=2, keepdim=True)

            with torch.no_grad():
                basis = self._estimate_basis(hd_centered, stage_idx=stage_idx).detach()

            k = int(basis.size(1))
            z_s = torch.einsum("ck,bcn->bkn", basis, s_centered).view(b, k, h, w)
            z_hd = torch.einsum("ck,bcn->bkn", basis, hd_centered).view(b, k, h, w)
            z_lr = torch.einsum("ck,bcn->bkn", basis, lr_centered).view(b, k, h, w)

            def rms_norm(z: torch.Tensor) -> torch.Tensor:
                z_flat = z.reshape(b, k, n)
                rms = z_flat.pow(2).mean(dim=2, keepdim=True).sqrt()
                return (z_flat / (rms + self.eps)).view(b, k, h, w)

            z_s = rms_norm(z_s)
            z_hd = rms_norm(z_hd)
            z_lr = rms_norm(z_lr)

            mask, csf_diag = self.csf_modules[key](
                f_hd=f_hd.float().detach(),
                z_hd=z_hd.detach(),
                z_lr=z_lr.detach(),
                depth=depth,
            )

            sq_error = (z_s - z_hd).pow(2)
            unweighted_mse = sq_error.mean()
            weighted_mse = sq_error * mask
            loss = weighted_mse.sum() / (mask.sum() + self.eps)
            coord_weights = mask.mean(dim=(0, 2, 3))
            coord_gap_abs = (z_hd - z_lr).abs()

            diagnostics = {
                "selected_k": z_hd.new_tensor(float(k)),
                "weight_mean": mask.mean(),
                "weight_std": mask.std(unbiased=False),
                "weight_min": mask.min(),
                "weight_max": mask.max(),
                "weight_pre_clamp_std": csf_diag["pre_clamp_mask_std"],
                "weight_pre_clamp_min": csf_diag["pre_clamp_mask_min"],
                "weight_pre_clamp_max": csf_diag["pre_clamp_mask_max"],
                "weight_low_clamp_frac": csf_diag["low_clamp_frac"],
                "weight_high_clamp_frac": csf_diag["high_clamp_frac"],
                "weighted_unweighted_ratio": loss.detach() / (unweighted_mse.detach() + self.eps),
                "unweighted_mse": unweighted_mse.detach(),
                "geometry_mean": csf_diag["csf_channel_mean"],
                "geometry_std": csf_diag["csf_channel_std"],
                "reach_mean": csf_diag["csf_spatial_mean"],
                "reach_std": csf_diag["csf_spatial_std"],
                "reach_min": mask.min(),
                "lr_scale_mean": csf_diag["lr_scale_mean"],
                "lr_scale_std": csf_diag["lr_scale_std"],
                "depth_gate_mean": csf_diag["depth_gate_mean"],
                "depth_gate_std": csf_diag["depth_gate_std"],
                "hd_lz_coord_mse": (z_hd - z_lr).pow(2).mean(),
                "hd_lz_coord_abs_mean": coord_gap_abs.mean(),
                "hd_lz_coord_abs_std": coord_gap_abs.std(unbiased=False),
                "coord_weight_std": coord_weights.std(unbiased=False),
                "coord_weight_min": coord_weights.min(),
                "coord_weight_max": coord_weights.max(),
                "depth_sim_mean": csf_diag["depth_sim_mean"],
                "depth_sim_std": csf_diag["depth_sim_std"],
            }
            return loss, diagnostics


class PrivilegedReachableGeoDecStageKSGSCEngine(PrivilegedReachableGeoDecSGSCEngine):
    """
    priv_reach_geo_dec_sgsc with explicit encoder stage-wise fixed k.

    Decoder KD keeps the original fixed k_decoder. Encoder KD uses k_stage_by_stage,
    so stages 0/1/2/3 can be swept independently without changing the decoder branch.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student, **kwargs)

        teacher_ch = kwargs.get("teacher_channels", [64, 128, 320, 512])
        self.k_stage_by_stage = _parse_stage_k(
            kwargs.get("k_stage_by_stage", None),
            default_k=int(kwargs.get("k_stage", 64)),
            num_stages=len(teacher_ch),
        )
        self.stage_sgsc_loss = StagewiseKReachableGeometryStageSGSCLoss(
            k_by_stage=self.k_stage_by_stage,
            default_k=int(kwargs.get("k_stage", 64)),
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
