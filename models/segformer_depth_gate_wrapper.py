"""
SegFormerDepthGateWrapper — Lightweight depth-conditioned SegFormer.

All-stage gate+bias fusion only (no cross-attention).
Inherits from SegFormerWrapper, preserving all existing hooks and interfaces.

Forward contract:
  - forward(x, depth=depth)                -> logits
  - forward(x, depth=depth, is_feat=True)  -> (fused_feats, logits, embeds, dec_fuse)
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DATA
from .segformer_wrapper import SegFormerWrapper


# ────────────────────────────────────────────────────────────
# Building blocks
# ────────────────────────────────────────────────────────────

class _ConvGNAct(nn.Module):
    """Conv2d → GroupNorm → GELU (lightweight, no bias on conv)."""

    def __init__(self, in_ch: int, out_ch: int, ks: int = 3,
                 stride: int = 1, groups: int = 8):
        super().__init__()
        pad = ks // 2
        ng = max(1, min(groups, out_ch))
        while out_ch % ng != 0 and ng > 1:
            ng -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=pad, bias=False),
            nn.GroupNorm(ng, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthPyramidEncoder(nn.Module):
    """
    Lightweight CNN producing depth features at 4 SegFormer stage resolutions.
    Approximate spatial scales: 1/4, 1/8, 1/16, 1/32 of input.
    """

    def __init__(self, out_channels: Sequence[int], in_ch: int = 1,
                 groups: int = 8):
        super().__init__()
        assert len(out_channels) == 4
        c1, c2, c3, c4 = [int(c) for c in out_channels]
        mid = max(c1 // 2, 16)

        self.s1 = nn.Sequential(
            _ConvGNAct(in_ch, mid, ks=7, stride=2, groups=groups),
            _ConvGNAct(mid, c1, ks=3, stride=2, groups=groups),
        )
        self.s2 = nn.Sequential(
            _ConvGNAct(c1, c2, ks=3, stride=2, groups=groups),
            _ConvGNAct(c2, c2, ks=3, stride=1, groups=groups),
        )
        self.s3 = nn.Sequential(
            _ConvGNAct(c2, c3, ks=3, stride=2, groups=groups),
            _ConvGNAct(c3, c3, ks=3, stride=1, groups=groups),
        )
        self.s4 = nn.Sequential(
            _ConvGNAct(c3, c4, ks=3, stride=2, groups=groups),
            _ConvGNAct(c4, c4, ks=3, stride=1, groups=groups),
        )

    def forward(self, depth: torch.Tensor):
        z1 = self.s1(depth)
        z2 = self.s2(z1)
        z3 = self.s3(z2)
        z4 = self.s4(z3)
        return z1, z2, z3, z4


class DepthGateBias(nn.Module):
    """
    Depth-guided gate + bias modulation.

        f_out = f_rgb * (1 + alpha * gate(z_depth)) + beta * bias(z_depth)

    alpha, beta are learnable scalars initialised to small values
    so that the model starts near identity (f_out ≈ f_rgb).
    """

    def __init__(self, channels: int, groups: int = 8,
                 init_alpha: float = 0.1, init_beta: float = 0.1):
        super().__init__()
        ng = max(1, min(groups, channels))
        while channels % ng != 0 and ng > 1:
            ng -= 1

        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(ng, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.bias = nn.Conv2d(channels, channels, 1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        if depth.shape[-2:] != rgb.shape[-2:]:
            depth = F.interpolate(depth, size=rgb.shape[-2:],
                                  mode="bilinear", align_corners=False)
        g = self.gate(depth)
        b = self.bias(depth)
        out = rgb * (1.0 + self.alpha * g) + self.beta * b
        stats = {
            "gate_mean": g.mean().detach(),
            "gate_std": g.std(unbiased=False).detach(),
            "alpha": self.alpha.detach(),
            "beta": self.beta.detach(),
        }
        return out, stats


# ────────────────────────────────────────────────────────────
# Main wrapper
# ────────────────────────────────────────────────────────────

class SegFormerDepthGateWrapper(SegFormerWrapper):
    """
    SegFormer with all-stage depth gate+bias conditioning.

    Key design decisions vs. V5 (AllStageDepthFusionWrapper):
      1. Cross-attention removed — gate+bias only at every stage.
      2. alpha/beta initialised small (0.1) for near-identity start.
      3. All parent hooks (dec_fuse, patch_embed) are inherited.
    """

    def __init__(
        self,
        name: str,
        num_classes: int = DATA.NUM_CLASSES,
        depth_in_ch: int = 1,
        depth_groups: int = 8,
        init_alpha: float = 0.1,
        init_beta: float = 0.1,
        require_depth: bool = True,
    ):
        super().__init__(name=name, num_classes=num_classes)

        hs = list(getattr(self.model.config, "hidden_sizes", []))
        if len(hs) < 4:
            raise RuntimeError(
                f"hidden_sizes must have ≥4 entries, got {hs}")
        self.stage_channels = [int(c) for c in hs[-4:]]
        self.require_depth = bool(require_depth)

        # depth encoder
        self.depth_encoder = DepthPyramidEncoder(
            out_channels=self.stage_channels,
            in_ch=int(depth_in_ch),
            groups=depth_groups,
        )

        # per-stage gate+bias (no cross-attention)
        self.fusion_blocks = nn.ModuleList([
            DepthGateBias(ch, groups=depth_groups,
                          init_alpha=init_alpha, init_beta=init_beta)
            for ch in self.stage_channels
        ])

        self._last_fusion_stats: dict[str, torch.Tensor] = {}

    # ── helpers ──────────────────────────────────────────────

    def get_last_fusion_stats(self) -> dict[str, torch.Tensor]:
        return dict(self._last_fusion_stats)

    def _prepare_depth(self, depth: Optional[torch.Tensor],
                       x: torch.Tensor) -> torch.Tensor:
        if depth is None:
            if self.require_depth:
                raise RuntimeError(
                    "SegFormerDepthGateWrapper requires depth=... in forward().")
            depth = torch.zeros(x.size(0), 1, x.size(2), x.size(3),
                                dtype=x.dtype, device=x.device)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.size(1) > 1:
            depth = depth[:, :1]
        depth = depth.to(device=x.device, dtype=x.dtype)
        if depth.shape[-2:] != x.shape[-2:]:
            depth = F.interpolate(depth, size=x.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return depth

    def _encode_rgb(self, x: torch.Tensor):
        rgb_out = self.model.segformer(
            pixel_values=x, output_hidden_states=True, return_dict=True)
        feats, embeds = self._collect_encoder_feats_and_embeds(
            rgb_out, x.shape[-2:])
        if feats is None or embeds is None:
            raise RuntimeError("Failed to obtain SegFormer hidden_states.")
        return feats, embeds

    def _fuse_stages(self, rgb_feats, depth_feats):
        fused, stats = [], {}
        for i, (rf, df, blk) in enumerate(
                zip(rgb_feats, depth_feats, self.fusion_blocks)):
            f, s = blk(rf, df)
            fused.append(f)
            for k, v in s.items():
                stats[f"{k}_s{i}"] = v
        return tuple(fused), stats

    # ── forward ──────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        return_feats: bool = False,
        return_embeds: bool = False,
        is_feat: bool = False,
    ):
        # clear caches
        for i in range(4):
            self._patch_embed_cache[i] = None
            self._patch_hw_cache[i] = None
        self._last_decoder_fuse = None
        self._last_fusion_stats = {}

        depth = self._prepare_depth(depth, x)

        # RGB encoder (SegFormer)
        rgb_feats, embeds = self._encode_rgb(x)

        # Depth encoder (lightweight CNN pyramid)
        depth_feats = self.depth_encoder(depth)

        # Per-stage gate+bias fusion
        fused_feats, fusion_stats = self._fuse_stages(rgb_feats, depth_feats)
        self._last_encoder_feats = fused_feats
        self._last_encoder_embeds = embeds
        self._last_fusion_stats = fusion_stats

        # Decode: HF decode_head expects tuple of encoder features
        # Hook on linear_fuse captures dec_fuse automatically
        logits_low = self.model.decode_head(fused_feats)
        logits = F.interpolate(logits_low, size=x.shape[-2:],
                               mode="bilinear", align_corners=False)

        if is_feat:
            dec_fuse = self._last_decoder_fuse
            return fused_feats, logits, embeds, dec_fuse

        if not return_feats and not return_embeds:
            return logits
        if return_feats and not return_embeds:
            return logits, fused_feats
        if not return_feats and return_embeds:
            return logits, embeds
        return logits, fused_feats, embeds
