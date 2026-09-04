from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .privileged_hierarchical_sgsc import PrivilegedHierarchicalSGSCEngine
from .privileged_transfer_sgsc import _autocast_off, _interpolate_antialias


def _safe_gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = max(1, min(int(max_groups), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DepthSimilarityCSF(nn.Module):
    """
    CSF-style selector for encoder KD.

    It receives the teacher HR+depth feature, the teacher HR/LR features
    projected into the HR+depth teacher eigen-coordinate space, and a
    pseudo-depth map. It predicts a coordinate-spatial KD mask from an
    LR-coordinate-scale-conditioned, geometry-modulated coordinate-gap stream
    and a teacher semantic-context stream, not from depth edges.
    """

    def __init__(
        self,
        channels: int,
        k: int,
        hidden_ch: int = 64,
        reduction: int = 4,
        mask_min: float = 0.25,
        mask_max: float = 4.0,
        lr_scale_alpha: float = 1.0,
        depth_gate_alpha: float = 1.0,
        depth_similarity_power: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.k = min(int(k), int(channels))
        self.mask_min = float(mask_min)
        self.mask_max = float(mask_max)
        self.lr_scale_alpha = float(lr_scale_alpha)
        self.depth_gate_alpha = float(depth_gate_alpha)
        self.depth_similarity_power = float(depth_similarity_power)
        self.eps = float(eps)

        hidden_ch = int(hidden_ch)
        mid_ch = max(hidden_ch // int(max(1, reduction)), 8)

        def proj(in_ch: int, kernel_size: int = 1) -> nn.Sequential:
            padding = kernel_size // 2
            return nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size, padding=padding, bias=False),
                _safe_gn(hidden_ch),
                nn.GELU(),
            )

        self.context_proj = proj(channels)
        self.diff_proj = proj(self.k)
        self.lr_scale_proj = nn.Sequential(
            nn.Conv2d(self.k, hidden_ch, kernel_size=1, bias=False),
            _safe_gn(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.depth_proj = proj(8, kernel_size=3)
        self.depth_gate_net = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch, kernel_size=3, padding=1, bias=False),
            _safe_gn(hidden_ch),
            nn.GELU(),
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_ch, mid_ch, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid_ch, self.k, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            _safe_gn(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def _local_depth_similarity(self, depth: torch.Tensor) -> torch.Tensor:
        b, _, h, w = depth.shape
        if h < 2 or w < 2:
            return depth.new_ones(b, 8, h, w)

        patches = F.unfold(depth, kernel_size=3, padding=1).view(b, 1, 9, h, w)
        center = patches[:, :, 4:5]
        neigh = torch.cat([patches[:, :, :4], patches[:, :, 5:]], dim=2)
        delta = (center - neigh).abs().squeeze(1)  # (B, 8, H, W)
        denom = delta.mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        norm_delta = delta / denom
        if self.depth_similarity_power != 1.0:
            norm_delta = norm_delta.clamp_min(0.0).pow(self.depth_similarity_power)
        return torch.exp(-norm_delta)

    def _depth_similarity_features(self, depth: torch.Tensor, size_hw: tuple[int, int]):
        h, w = size_hw
        depth = F.interpolate(depth, size=size_hw, mode="bilinear", align_corners=False)
        depth_feats = []
        sim_means = []
        sim_stds = []
        for scale in (1, 2, 4):
            if scale > 1:
                if h < scale * 2 or w < scale * 2:
                    continue
                depth_s = F.avg_pool2d(depth, kernel_size=scale, stride=scale, ceil_mode=False)
            else:
                depth_s = depth

            sim = self._local_depth_similarity(depth_s)
            feat = self.depth_proj(sim)
            if feat.shape[-2:] != size_hw:
                feat = F.interpolate(feat, size=size_hw, mode="bilinear", align_corners=False)
            depth_feats.append(feat)
            sim_means.append(sim.mean())
            sim_stds.append(sim.std(unbiased=False))

        depth_feat = torch.stack(depth_feats, dim=0).mean(dim=0)
        sim_mean = torch.stack(sim_means).mean()
        sim_std = torch.stack(sim_stds).mean()
        return depth_feat, sim_mean.detach(), sim_std.detach()

    def forward(
        self,
        f_hd: torch.Tensor,
        z_hd: torch.Tensor,
        z_lr: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if z_hd.shape != z_lr.shape:
            raise RuntimeError(
                f"DS-CSF coordinate shapes must match, got {tuple(z_hd.shape)} vs "
                f"{tuple(z_lr.shape)}"
            )
        if f_hd.shape[0] != z_hd.shape[0] or f_hd.shape[-2:] != z_hd.shape[-2:]:
            raise RuntimeError(
                "DS-CSF context and coordinate shapes must share batch/spatial size, got "
                f"context={tuple(f_hd.shape)}, coord={tuple(z_hd.shape)}"
            )

        size_hw = z_hd.shape[-2:]
        diff = (z_hd - z_lr).abs()
        depth_feat, sim_mean, sim_std = self._depth_similarity_features(depth, size_hw)
        diff_feat = self.diff_proj(diff)
        lr_scale = self.lr_scale_proj(z_lr)
        depth_gate = self.depth_gate_net(depth_feat)
        modulated_diff = (
            diff_feat
            * (1.0 + self.lr_scale_alpha * lr_scale)
            * (1.0 + self.depth_gate_alpha * depth_gate)
        )
        context = self.context_proj(f_hd)

        q = torch.cat(
            [
                modulated_diff,
                context,
            ],
            dim=1,
        )
        h = self.fuse(q)
        channel = self.channel_attn(h)
        spatial = self.spatial_attn(h)
        raw_mask = channel * spatial
        pre_clamp_mask = raw_mask / (raw_mask.mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        low_clamp_frac = (pre_clamp_mask <= self.mask_min).float().mean()
        high_clamp_frac = (pre_clamp_mask >= self.mask_max).float().mean()
        mask = pre_clamp_mask.clamp(self.mask_min, self.mask_max)
        mask = mask / (mask.mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        diag = {
            "csf_channel_mean": channel.detach().mean(),
            "csf_channel_std": channel.detach().std(unbiased=False),
            "csf_spatial_mean": spatial.detach().mean(),
            "csf_spatial_std": spatial.detach().std(unbiased=False),
            "raw_mask_mean": raw_mask.detach().mean(),
            "raw_mask_std": raw_mask.detach().std(unbiased=False),
            "pre_clamp_mask_std": pre_clamp_mask.detach().std(unbiased=False),
            "pre_clamp_mask_min": pre_clamp_mask.detach().min(),
            "pre_clamp_mask_max": pre_clamp_mask.detach().max(),
            "low_clamp_frac": low_clamp_frac.detach(),
            "high_clamp_frac": high_clamp_frac.detach(),
            "lr_scale_mean": lr_scale.detach().mean(),
            "lr_scale_std": lr_scale.detach().std(unbiased=False),
            "depth_gate_mean": depth_gate.detach().mean(),
            "depth_gate_std": depth_gate.detach().std(unbiased=False),
            "depth_sim_mean": sim_mean,
            "depth_sim_std": sim_std,
        }
        return mask, diag


class ReachableGeometryStageSGSCLoss(nn.Module):
    """
    Encoder-stage SGSC with learnable depth-similarity CSF weighting.

    The spectral basis and target coordinates come from the teacher HR+depth
    feature. The KD weight is predicted from a depth-similarity-modulated
    HR/LR reachability gap, LR activation-scale context, and teacher HR+depth
    semantic context.
    """

    def __init__(
        self,
        k: int = 64,
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
        super().__init__()
        self.k = int(k)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.eps = float(eps)
        self.csf_modules = nn.ModuleDict()
        if stage_channels is not None:
            for idx, ch in enumerate(stage_channels):
                self.csf_modules[str(idx)] = DepthSimilarityCSF(
                    channels=int(ch),
                    k=self.k,
                    hidden_ch=csf_hidden_ch,
                    reduction=csf_reduction,
                    mask_min=weight_min,
                    mask_max=weight_max,
                    lr_scale_alpha=lr_scale_alpha,
                    depth_gate_alpha=depth_gate_alpha,
                    depth_similarity_power=depth_similarity_power,
                    eps=eps,
                )

    def _estimate_basis(self, t_centered: torch.Tensor) -> torch.Tensor:
        with _autocast_off(t_centered.device):
            t_c = t_centered.float()
            b, c, n = t_c.shape
            k = min(self.k, c)
            cov = torch.einsum("bcn,bdn->cd", t_c, t_c) / (float(b * n) + self.eps)
            _, eigenvectors = torch.linalg.eigh(cov)
            return eigenvectors[:, -k:]

    def _prepare_depth(self, depth: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        if not torch.is_tensor(depth):
            raise RuntimeError(
                "ReachableGeometryStageSGSCLoss requires depth in imgs=(x_lr,x_hr,depth)."
            )
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() != 4:
            raise RuntimeError(f"depth must be (B,H,W) or (B,1,H,W), got {tuple(depth.shape)}")
        if depth.size(1) != 1:
            depth = depth[:, :1]
        depth = depth.float()
        if depth.shape[-2:] != size_hw:
            depth = F.interpolate(depth, size=size_hw, mode="bilinear", align_corners=False)
        return depth

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
                "Reachable geometry SGSC feature shapes must match, got "
                f"student={tuple(f_s.shape)}, hr_depth={tuple(f_hd.shape)}, "
                f"lr_rgb={tuple(f_lr_rgb.shape)}"
            )
        key = str(int(stage_idx))
        if key not in self.csf_modules:
            raise RuntimeError(
                f"DS-CSF module for stage {stage_idx} is not initialized. "
                "Pass teacher stage channels when constructing ReachableGeometryStageSGSCLoss."
            )

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
                basis = self._estimate_basis(hd_centered).detach()

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


class PrivilegedReachableGeoSGSCEngine(PrivilegedHierarchicalSGSCEngine):
    """
    Encoder-only privileged KD.

    Student:
        x_lr

    Teacher:
        x_hr + depth     -> target and spectral basis
        x_lr RGB-only    -> LR-domain reachability anchor

    Decoder transfer KD is intentionally disabled here so this engine can be
    used as a clean encoder contribution experiment.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student, **kwargs)
        self.lambda_transfer = 0.0
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

    def get_extra_parameters(self):
        params = list(super().get_extra_parameters())
        params.extend(self.stage_sgsc_loss.parameters())
        return params

    @staticmethod
    def _zero_depth_like(x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.size(0), 1, x.size(2), x.size(3))

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

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        x_lr, x_hr, depth = self._unpack_inputs(imgs)
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        if masks.dim() != 3:
            raise RuntimeError(f"masks must be (B,H,W), got {tuple(masks.shape)}")

        s_out = self.student(x_lr, is_feat=True)
        s_feats, s_logits, _dec_fuse_s = self._extract_feats_logits_decfuse(s_out, "student")

        with torch.no_grad():
            self.teacher.eval()
            t_hd_out = self.teacher(x_hr, depth=depth, is_feat=True)
            t_hd_feats, _t_logits_unused, _dec_fuse_unused = self._extract_feats_logits_decfuse(
                t_hd_out, "teacher_hr_depth"
            )
            t_lz_out = self._teacher_lr_rgb_only_forward(x_lr)
            t_lz_feats, _t_lz_logits_unused, _dec_lz_unused = self._extract_feats_logits_decfuse(
                t_lz_out, "teacher_lr_rgb_only"
            )

        ce = self.ce(s_logits, masks) * self.w_ce_student
        stage_sum = torch.tensor(0.0, device=device)
        stage_weight_sum = 0.0
        stage_loss_dict: Dict[str, torch.Tensor] = {}
        weight_means = []
        weight_stds = []
        weight_maxes = []
        geometry_stds = []
        reach_stds = []
        reach_mins = []
        hd_lz_mses = []

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
            weight_maxes.append(diag["weight_max"].detach())
            geometry_stds.append(diag["geometry_std"].detach())
            reach_stds.append(diag["reach_std"].detach())
            reach_mins.append(diag["reach_min"].detach())
            hd_lz_mses.append(diag["hd_lz_coord_mse"].detach())

        if stage_weight_sum > 0:
            stage_raw = stage_sum / stage_weight_sum
        else:
            stage_raw = torch.tensor(0.0, device=device)
        stage_kd = stage_raw * self.lambda_stage
        total = ce + stage_kd
        student_miou, student_pa = self._seg_metrics(s_logits, masks)

        out: Dict[str, Any] = {
            "total": total,
            "ce_student": ce.detach(),
            "reachable_geo_stage_sgsc": stage_kd.detach(),
            "reachable_geo_stage_sgsc_raw": stage_raw.detach(),
            "student_mIoU": student_miou.detach(),
            "student_pixel_acc": student_pa.detach(),
            "s_logits": s_logits.detach(),
            "student_input": x_lr.detach(),
            "teacher_input": x_hr.detach(),
        }
        if weight_means:
            out["rg_weight_mean"] = torch.stack(weight_means).mean()
            out["rg_weight_std"] = torch.stack(weight_stds).mean()
            out["rg_weight_max"] = torch.stack(weight_maxes).max()
            out["rg_geometry_std"] = torch.stack(geometry_stds).mean()
            out["rg_reach_std"] = torch.stack(reach_stds).mean()
            out["rg_reach_min"] = torch.stack(reach_mins).min()
            out["rg_hd_lz_coord_mse"] = torch.stack(hd_lz_mses).mean()
        out.update(stage_loss_dict)
        return out
