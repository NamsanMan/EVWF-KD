"""
SegFormer feature-discrepancy conditioned suppression teacher.

This is a GBST variant for teacher preparation.  The privileged HR path uses
the degraded LR view as a conditioning signal for the suppression gate, but
never injects the degraded LR feature into the output feature itself.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DATA
from .segformer_depth_gate_wrapper import DepthPyramidEncoder
from .segformer_wrapper import SegFormerWrapper


def _gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = max(1, min(int(max_groups), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _inv_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(value)).item())


class FDCSSuppressionBlock(nn.Module):
    """
    Feature-discrepancy conditioned RGB suppression and geometry injection.

        delta = |F_hr - F_lr| / (|F_hr| + |F_lr| + eps)
        g = sigmoid(phi([F_hr, F_geo, delta]))
        F_out = F_hr * (1 - alpha * g) + beta * F_geo * g

    delta is an input to the gate only; it does not appear in F_out.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        init_alpha: float = 0.1,
        init_beta: float = 0.1,
        max_alpha: float = 0.75,
        max_beta: float = 1.0,
        normalize_delta: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        channels = int(channels)
        self.max_alpha = float(max_alpha)
        self.max_beta = float(max_beta)
        self.normalize_delta = bool(normalize_delta)
        self.eps = float(eps)

        self.depth_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _gn(channels, groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            _gn(channels, groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        alpha_ratio = float(init_alpha) / max(self.max_alpha, 1e-6)
        beta_ratio = float(init_beta) / max(self.max_beta, 1e-6)
        self.alpha_logit = nn.Parameter(torch.tensor(_inv_sigmoid(alpha_ratio)))
        self.beta_logit = nn.Parameter(torch.tensor(_inv_sigmoid(beta_ratio)))

    def _feature_delta(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        if lr.shape[-2:] != hr.shape[-2:]:
            lr = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
        delta = (hr - lr).abs()
        if self.normalize_delta:
            denom = hr.abs() + lr.abs() + self.eps
            delta = delta / denom
        return delta

    def forward(
        self,
        rgb_hr: torch.Tensor,
        depth: torch.Tensor,
        rgb_lr: Optional[torch.Tensor] = None,
    ):
        if depth.shape[-2:] != rgb_hr.shape[-2:]:
            depth = F.interpolate(depth, size=rgb_hr.shape[-2:], mode="bilinear", align_corners=False)
        if rgb_lr is None:
            delta = torch.zeros_like(rgb_hr)
        else:
            delta = self._feature_delta(rgb_hr, rgb_lr)

        geo = self.depth_proj(depth)
        gate = self.gate(torch.cat([rgb_hr, geo, delta], dim=1))
        alpha = torch.sigmoid(self.alpha_logit) * self.max_alpha
        beta = torch.sigmoid(self.beta_logit) * self.max_beta
        out = rgb_hr * (1.0 - alpha * gate) + beta * geo * gate

        stats = {
            "suppress_gate_mean": gate.mean().detach(),
            "suppress_gate_std": gate.std(unbiased=False).detach(),
            "suppress_alpha": alpha.detach(),
            "geo_beta": beta.detach(),
            "geo_mean_abs": geo.abs().mean().detach(),
            "geo_std": geo.std(unbiased=False).detach(),
            "fdcs_delta_mean": delta.mean().detach(),
            "fdcs_delta_std": delta.std(unbiased=False).detach(),
        }
        return out, stats


class SegFormerFDCSTeacherWrapper(SegFormerWrapper):
    """
    SegFormer teacher with feature-discrepancy conditioned suppression.

    Forward contract:
      - forward(x, depth=depth, lr_condition=x_lr) -> logits
      - forward(..., is_feat=True) -> (feats, logits, embeds, dec_fuse)
    """

    supports_lr_condition = True

    def __init__(
        self,
        name: str,
        num_classes: int = DATA.NUM_CLASSES,
        depth_in_ch: int = 1,
        depth_groups: int = 8,
        init_alpha: float = 0.1,
        init_beta: float = 0.1,
        max_alpha: float = 0.75,
        max_beta: float = 1.0,
        normalize_delta: bool = True,
        require_depth: bool = True,
    ):
        super().__init__(name=name, num_classes=num_classes)

        hs = list(getattr(self.model.config, "hidden_sizes", []))
        if len(hs) < 4:
            raise RuntimeError(f"hidden_sizes must have at least 4 entries, got {hs}")
        self.stage_channels = [int(c) for c in hs[-4:]]
        self.require_depth = bool(require_depth)

        self.depth_encoder = DepthPyramidEncoder(
            out_channels=self.stage_channels,
            in_ch=int(depth_in_ch),
            groups=depth_groups,
        )
        self.suppression_blocks = nn.ModuleList(
            [
                FDCSSuppressionBlock(
                    channels=ch,
                    groups=depth_groups,
                    init_alpha=init_alpha,
                    init_beta=init_beta,
                    max_alpha=max_alpha,
                    max_beta=max_beta,
                    normalize_delta=normalize_delta,
                )
                for ch in self.stage_channels
            ]
        )
        self._last_fusion_stats: dict[str, torch.Tensor] = {}

    def get_last_fusion_stats(self) -> dict[str, torch.Tensor]:
        return dict(self._last_fusion_stats)

    def _prepare_depth(self, depth: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if depth is None:
            if self.require_depth:
                raise RuntimeError("SegFormerFDCSTeacherWrapper requires depth=... in forward().")
            depth = torch.zeros(x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.size(1) > 1:
            depth = depth[:, :1]
        depth = depth.to(device=x.device, dtype=x.dtype)
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return depth

    def _encode_rgb(self, x: torch.Tensor):
        rgb_out = self.model.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        feats, embeds = self._collect_encoder_feats_and_embeds(rgb_out, x.shape[-2:])
        if feats is None or embeds is None:
            raise RuntimeError("Failed to obtain SegFormer hidden states.")
        return feats, embeds

    def _suppress_stages(self, hr_feats, depth_feats, lr_feats=None):
        suppressed, stats = [], {}
        lr_iter = lr_feats if lr_feats is not None else [None] * len(hr_feats)
        for i, (hf, df, lf, block) in enumerate(
            zip(hr_feats, depth_feats, lr_iter, self.suppression_blocks)
        ):
            feat, stage_stats = block(hf, df, lf)
            suppressed.append(feat)
            for key, value in stage_stats.items():
                stats[f"{key}_s{i}"] = value
        return tuple(suppressed), stats

    def forward(
        self,
        x: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        lr_condition: Optional[torch.Tensor] = None,
        return_feats: bool = False,
        return_embeds: bool = False,
        is_feat: bool = False,
    ):
        for i in range(4):
            self._patch_embed_cache[i] = None
            self._patch_hw_cache[i] = None
        self._last_decoder_fuse = None
        self._last_fusion_stats = {}

        depth = self._prepare_depth(depth, x)
        rgb_feats, embeds = self._encode_rgb(x)

        use_geometry = bool(depth.detach().abs().sum().item() > 0.0)
        if use_geometry:
            lr_feats = None
            if lr_condition is not None:
                lr_condition = lr_condition.to(device=x.device, dtype=x.dtype)
                if lr_condition.shape[-2:] != x.shape[-2:]:
                    lr_condition = F.interpolate(
                        lr_condition, size=x.shape[-2:], mode="bilinear", align_corners=False
                    )
                lr_feats, _ = self._encode_rgb(lr_condition)

            depth_feats = self.depth_encoder(depth)
            feats, stats = self._suppress_stages(rgb_feats, depth_feats, lr_feats)
            for i in range(4):
                stats[f"geometry_skipped_s{i}"] = depth.new_tensor(0.0)
                stats[f"fdcs_condition_used_s{i}"] = depth.new_tensor(float(lr_feats is not None))
        else:
            feats = tuple(rgb_feats)
            stats = {}
            for i, block in enumerate(self.suppression_blocks):
                alpha = torch.sigmoid(block.alpha_logit) * block.max_alpha
                beta = torch.sigmoid(block.beta_logit) * block.max_beta
                stats[f"suppress_gate_mean_s{i}"] = depth.new_tensor(0.0)
                stats[f"suppress_gate_std_s{i}"] = depth.new_tensor(0.0)
                stats[f"suppress_alpha_s{i}"] = alpha.detach()
                stats[f"geo_beta_s{i}"] = beta.detach()
                stats[f"geo_mean_abs_s{i}"] = depth.new_tensor(0.0)
                stats[f"geo_std_s{i}"] = depth.new_tensor(0.0)
                stats[f"fdcs_delta_mean_s{i}"] = depth.new_tensor(0.0)
                stats[f"fdcs_delta_std_s{i}"] = depth.new_tensor(0.0)
                stats[f"geometry_skipped_s{i}"] = depth.new_tensor(1.0)
                stats[f"fdcs_condition_used_s{i}"] = depth.new_tensor(0.0)

        self._last_encoder_feats = feats
        self._last_encoder_embeds = embeds
        self._last_fusion_stats = stats

        logits_low = self.model.decode_head(feats)
        logits = F.interpolate(logits_low, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if is_feat:
            return feats, logits, embeds, self._last_decoder_fuse
        if not return_feats and not return_embeds:
            return logits
        if return_feats and not return_embeds:
            return logits, feats
        if not return_feats and return_embeds:
            return logits, embeds
        return logits, feats, embeds
