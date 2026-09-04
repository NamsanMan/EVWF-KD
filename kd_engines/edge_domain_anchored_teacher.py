from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from .base_engine import BaseKDEngine


class EdgeDomainAnchoredTeacherEngine(BaseKDEngine):
    """
    Teacher-only training for an edge-device degraded LR deployment domain.

    The teacher's privileged path remains clean HR + pseudo depth. A second
    degraded-LR path is used only during teacher preparation to anchor the
    teacher to the student deployment domain. No decoder G/private split is
    introduced.

        HR path: T(x_hr, depth)
        LR path: T(x_lr, depth_anchor)

        L = CE_hr + w_ce_lr * CE_lr + w_rel * Rel(HR, LR; depth edges)

    `depth_anchor` can be:
        - "zero"    : LR path receives zero depth, closest to RGB-only student.
        - "full"    : LR path also receives depth.
        - "dropout" : sample-wise depth dropout, a compromise between both.
    """

    trains_teacher_only = True
    primary_eval_view = "lr"

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student)

        self.w_ce_hr = float(kwargs.get("w_ce_hr", 1.0))
        self.w_ce_lr = float(kwargs.get("w_ce_lr", 0.5))
        self.w_rel = float(kwargs.get("w_rel", 0.25))

        self.rel_target = str(kwargs.get("rel_target", "dec_fuse"))
        self.rel_detach_hr = bool(kwargs.get("rel_detach_hr", True))
        self.use_depth_weight = bool(kwargs.get("use_depth_weight", True))
        self.depth_weight_lambda = float(kwargs.get("depth_weight_lambda", 2.0))

        self.lr_depth_mode = str(kwargs.get("lr_depth_mode", "dropout")).lower()
        self.lr_depth_keep_prob = float(kwargs.get("lr_depth_keep_prob", 0.3))
        if self.lr_depth_mode not in {"zero", "full", "dropout"}:
            raise ValueError(
                f"lr_depth_mode must be one of zero/full/dropout, got {self.lr_depth_mode}"
            )
        if not (0.0 <= self.lr_depth_keep_prob <= 1.0):
            raise ValueError("lr_depth_keep_prob must be in [0, 1].")

        self.eval_depth_mode = str(kwargs.get("eval_depth_mode", "zero")).lower()
        if self.eval_depth_mode not in {"input", "zero"}:
            raise ValueError("eval_depth_mode must be 'input' or 'zero'.")

        self.ignore_index = int(
            kwargs.get("ignore_index", getattr(config.DATA, "IGNORE_INDEX", 255))
        )
        self.num_classes = int(kwargs.get("num_classes", getattr(config.DATA, "NUM_CLASSES", 0)))
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def get_primary_model(self) -> nn.Module:
        return self.teacher

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        return []

    @staticmethod
    def _extract_output(model_out: Any):
        if not isinstance(model_out, (tuple, list)) or len(model_out) < 4:
            raise RuntimeError(
                "Wrapper forward(x, depth=..., is_feat=True) must return "
                "(feats, logits, embeds, dec_fuse)."
            )
        feats = model_out[0]
        logits = model_out[1]
        dec_fuse = model_out[3]
        return logits, feats, dec_fuse

    @staticmethod
    def _compute_local_relations(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        feat = F.normalize(feat, p=2, dim=1, eps=1e-6)
        patches = F.unfold(feat, kernel_size=3, padding=1).view(b, c, 9, h, w)
        center = patches[:, :, 4:5]
        neigh = torch.cat([patches[:, :, :4], patches[:, :, 5:]], dim=2)
        return (center * neigh).sum(dim=1)

    @staticmethod
    def _depth_gradient_magnitude(depth: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=depth.dtype,
            device=depth.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=depth.dtype,
            device=depth.device,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(depth, sobel_x, padding=1)
        gy = F.conv2d(depth, sobel_y, padding=1)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)
        b = mag.size(0)
        flat = mag.flatten(1)
        max_val = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1)
        return mag / (max_val + 1e-8)

    def _anchor_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if self.lr_depth_mode == "full":
            return depth
        if self.lr_depth_mode == "zero":
            return torch.zeros_like(depth)

        # Sample-wise dropout. Kept depth samples preserve geometric guidance;
        # zero-depth samples keep the anchor close to RGB-only deployment.
        if self.training:
            keep = (
                torch.rand(depth.size(0), 1, 1, 1, device=depth.device, dtype=depth.dtype)
                < self.lr_depth_keep_prob
            ).to(depth.dtype)
            return depth * keep
        return depth * float(self.lr_depth_keep_prob)

    def _get_rel_features(self, feats_hr, feats_lr, dec_fuse_hr, dec_fuse_lr):
        if self.rel_target == "dec_fuse":
            if dec_fuse_hr is None or dec_fuse_lr is None:
                raise RuntimeError("rel_target='dec_fuse' but dec_fuse is None.")
            return dec_fuse_hr, dec_fuse_lr
        if self.rel_target == "encoder":
            return feats_hr[-1], feats_lr[-1]
        raise ValueError(f"Unknown rel_target: {self.rel_target}")

    def _compute_relation_loss(
        self,
        feat_hr: torch.Tensor,
        feat_lr: torch.Tensor,
        depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_hw = feat_lr.shape[-2:]
        if feat_hr.shape[-2:] != target_hw:
            feat_hr = F.interpolate(feat_hr, size=target_hw, mode="bilinear", align_corners=False)

        rel_hr = self._compute_local_relations(feat_hr)
        rel_lr = self._compute_local_relations(feat_lr)
        if self.rel_detach_hr:
            rel_hr = rel_hr.detach()

        raw = (rel_lr - rel_hr).abs()
        weighted = raw
        depth_edge_mean = raw.new_tensor(0.0)
        if self.use_depth_weight:
            depth_resized = F.interpolate(depth, size=target_hw, mode="bilinear", align_corners=False)
            grad_mag = self._depth_gradient_magnitude(depth_resized)
            depth_edge_mean = grad_mag.mean().detach()
            weighted = raw * (1.0 + self.depth_weight_lambda * grad_mag)

        return {
            "rel_loss": weighted.mean(),
            "rel_raw_mean": raw.mean().detach(),
            "rel_weighted_mean": weighted.mean().detach(),
            "depth_edge_mean": depth_edge_mean,
        }

    def _seg_metrics(self, logits: torch.Tensor, masks: torch.Tensor):
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            valid = masks != self.ignore_index
            if not valid.any():
                z = logits.new_tensor(0.0)
                return z, z
            pred_v = pred[valid].clamp(0, self.num_classes - 1)
            mask_v = masks[valid].clamp(0, self.num_classes - 1)
            pixel_acc = (pred_v == mask_v).float().mean()
            idx = mask_v * self.num_classes + pred_v
            conf = torch.bincount(idx, minlength=self.num_classes * self.num_classes)
            conf = conf.reshape(self.num_classes, self.num_classes).float()
            inter = conf.diag()
            union = conf.sum(dim=1) + conf.sum(dim=0) - inter
            valid_cls = union > 0
            miou = torch.where(valid_cls, inter / union.clamp_min(1.0), torch.zeros_like(union))
            miou = miou[valid_cls].mean() if valid_cls.any() else logits.new_tensor(0.0)
            return miou, pixel_acc

    @staticmethod
    def _match_logits(logits: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        if logits.shape[-2:] == size_hw:
            return logits
        return F.interpolate(logits, size=size_hw, mode="bilinear", align_corners=False)

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        if not isinstance(imgs, (tuple, list)) or len(imgs) < 3:
            raise RuntimeError("EdgeDomainAnchoredTeacherEngine requires imgs=(x_LR, x_HR, depth).")

        x_lr, x_hr, depth = imgs[:3]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.dim() != 4:
            raise RuntimeError(f"depth must be (B,H,W) or (B,1,H,W), got {tuple(depth.shape)}")
        if depth.size(1) != 1:
            depth = depth[:, :1]

        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        if masks.dim() != 3:
            raise RuntimeError(f"masks must be (B,H,W), got {tuple(masks.shape)}")

        model = self.teacher
        depth_anchor = self._anchor_depth(depth)

        logits_hr, feats_hr, dec_fuse_hr = self._extract_output(
            model(x_hr, depth=depth, is_feat=True)
        )
        logits_lr, feats_lr, dec_fuse_lr = self._extract_output(
            model(x_lr, depth=depth_anchor, is_feat=True)
        )

        logits_hr_up = self._match_logits(logits_hr, masks.shape[-2:])
        logits_lr_up = self._match_logits(logits_lr, masks.shape[-2:])

        ce_hr = self.ce(logits_hr_up, masks) * self.w_ce_hr
        ce_lr = self.ce(logits_lr_up, masks) * self.w_ce_lr

        feat_hr_rel, feat_lr_rel = self._get_rel_features(
            feats_hr, feats_lr, dec_fuse_hr, dec_fuse_lr
        )
        rel_out = self._compute_relation_loss(feat_hr_rel, feat_lr_rel, depth)
        rel_loss = rel_out["rel_loss"] * self.w_rel
        total = ce_hr + ce_lr + rel_loss

        hr_miou, hr_pa = self._seg_metrics(logits_hr_up, masks)
        lr_miou, lr_pa = self._seg_metrics(logits_lr_up, masks)

        out: Dict[str, Any] = {
            "total": total,
            "ce_hr": ce_hr.detach(),
            "ce_lr": ce_lr.detach(),
            "rel_loss": rel_loss.detach(),
            "rel_raw_mean": rel_out["rel_raw_mean"],
            "rel_weighted_mean": rel_out["rel_weighted_mean"],
            "depth_edge_mean": rel_out["depth_edge_mean"],
            "hr_mIoU": hr_miou.detach(),
            "hr_pixel_acc": hr_pa.detach(),
            "lr_mIoU": lr_miou.detach(),
            "lr_pixel_acc": lr_pa.detach(),
            "lr_depth_keep_actual": (depth_anchor.abs().sum(dim=(1, 2, 3)) > 0).float().mean().detach(),
            "s_logits": logits_lr_up.detach(),
            "t_logits": logits_hr_up.detach(),
            "student_input": x_lr.detach(),
            "teacher_input": x_hr.detach(),
        }

        if hasattr(model, "get_last_fusion_stats"):
            stats = model.get_last_fusion_stats()
            if isinstance(stats, dict):
                for key, value in stats.items():
                    out[key] = value.detach() if torch.is_tensor(value) else value

        return out
