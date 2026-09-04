from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .edge_domain_anchored_teacher import EdgeDomainAnchoredTeacherEngine


class FDCSGBSTTeacherEngine(EdgeDomainAnchoredTeacherEngine):
    """
    Teacher-only training for FDCS-GBST.

        HR path: T(x_hr, depth, lr_condition=x_lr)
        LR path: T(x_lr, zero_depth)

        L = CE_HR + w_ce_lr * CE_LR + w_lapc * LAPC(dec_fuse_HR, dec_fuse_LR)

    LAPC aligns class-wise dec_fuse prototypes. Each class contribution is
    softly weighted by the degraded LR RGB-only path confidence on its ground
    truth region, avoiding hard observability thresholds.
    """

    trains_teacher_only = True
    primary_eval_view = "lr"

    def __init__(self, teacher: nn.Module, student: nn.Module, **kwargs):
        super().__init__(teacher, student, **kwargs)
        self.w_ce_hr = float(kwargs.get("w_ce_hr", 1.0))
        self.w_ce_lr = float(kwargs.get("w_ce_lr", 1.0))
        self.w_lapc = float(kwargs.get("w_lapc", 0.1))
        self.lr_depth_mode = "zero"
        self.eval_depth_mode = str(kwargs.get("eval_depth_mode", "zero")).lower()
        if self.eval_depth_mode not in {"input", "zero"}:
            raise ValueError("eval_depth_mode must be 'input' or 'zero'.")
        self.lapc_detach_lr = bool(kwargs.get("lapc_detach_lr", True))
        self.eps = float(kwargs.get("eps", 1e-6))

    def get_primary_model(self) -> nn.Module:
        return self.teacher

    def get_extra_parameters(self) -> Iterable[nn.Parameter]:
        return []

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _teacher_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        depth: torch.Tensor,
        lr_condition: torch.Tensor | None = None,
    ):
        inner = self._unwrap_model(model)
        if lr_condition is not None and getattr(inner, "supports_lr_condition", False):
            return model(x, depth=depth, lr_condition=lr_condition, is_feat=True)
        return model(x, depth=depth, is_feat=True)

    def _lapc_loss(
        self,
        dec_hr: torch.Tensor,
        dec_lr: torch.Tensor,
        logits_lr: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if dec_hr is None or dec_lr is None:
            raise RuntimeError("FDCS-GBST LAPC requires dec_fuse from both teacher paths.")
        if dec_hr.shape != dec_lr.shape:
            dec_hr = F.interpolate(
                dec_hr, size=dec_lr.shape[-2:], mode="bilinear", align_corners=False
            )

        b, ch, h, w = dec_lr.shape
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        masks_dec = F.interpolate(
            masks.unsqueeze(1).float(),
            size=(h, w),
            mode="nearest",
        ).squeeze(1).long()

        valid = masks_dec != self.ignore_index
        if not valid.any():
            z = dec_lr.new_tensor(0.0)
            return {
                "lapc": z,
                "lapc_class_count": z,
                "lapc_conf_mean": z,
                "lapc_proto_cos": z,
            }

        logits_lr_dec = F.interpolate(
            logits_lr.detach(), size=(h, w), mode="bilinear", align_corners=False
        )
        prob_lr = F.softmax(logits_lr_dec, dim=1).detach()

        labels = masks_dec.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).permute(0, 3, 1, 2)
        one_hot = one_hot.to(device=dec_lr.device, dtype=dec_lr.dtype)
        one_hot = one_hot * valid.unsqueeze(1).to(dtype=one_hot.dtype)

        mask_flat = one_hot.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        denom = mask_flat.sum(dim=0)
        class_valid = denom > 0
        if not class_valid.any():
            z = dec_lr.new_tensor(0.0)
            return {
                "lapc": z,
                "lapc_class_count": z,
                "lapc_conf_mean": z,
                "lapc_proto_cos": z,
            }

        hr_flat = dec_hr.permute(0, 2, 3, 1).reshape(-1, ch)
        lr_feat = dec_lr.detach() if self.lapc_detach_lr else dec_lr
        lr_flat = lr_feat.permute(0, 2, 3, 1).reshape(-1, ch)
        proto_hr = mask_flat.transpose(0, 1).matmul(hr_flat) / denom.clamp_min(self.eps).unsqueeze(1)
        proto_lr = mask_flat.transpose(0, 1).matmul(lr_flat) / denom.clamp_min(self.eps).unsqueeze(1)

        prob_flat = prob_lr.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        class_conf = (mask_flat * prob_flat).sum(dim=0) / denom.clamp_min(self.eps)
        class_conf = class_conf.detach()

        cos = F.cosine_similarity(proto_hr, proto_lr, dim=1, eps=self.eps)
        proto_loss = 1.0 - cos
        weights = class_conf * class_valid.to(dtype=class_conf.dtype)
        lapc = (proto_loss * weights).sum() / weights.sum().clamp_min(self.eps)

        valid_weights = weights[class_valid]
        valid_cos = cos[class_valid]
        return {
            "lapc": lapc,
            "lapc_class_count": class_valid.to(dec_lr.dtype).sum().detach(),
            "lapc_conf_mean": valid_weights.mean().detach(),
            "lapc_proto_cos": valid_cos.mean().detach(),
        }

    def compute_losses(self, imgs: Any, masks: torch.Tensor, device) -> Dict[str, Any]:
        if not isinstance(imgs, (tuple, list)) or len(imgs) < 3:
            raise RuntimeError("FDCSGBSTTeacherEngine requires imgs=(x_LR, x_HR, depth).")

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
        zero_depth = torch.zeros_like(depth)

        logits_hr, _feats_hr, dec_hr = self._extract_output(
            self._teacher_forward(model, x_hr, depth=depth, lr_condition=x_lr)
        )
        hr_stats = {}
        if hasattr(model, "get_last_fusion_stats"):
            stats = model.get_last_fusion_stats()
            if isinstance(stats, dict):
                hr_stats = {
                    f"hr_{key}": value.detach() if torch.is_tensor(value) else value
                    for key, value in stats.items()
                }

        logits_lr, _feats_lr, dec_lr = self._extract_output(
            self._teacher_forward(model, x_lr, depth=zero_depth, lr_condition=None)
        )
        lr_stats = {}
        if hasattr(model, "get_last_fusion_stats"):
            stats = model.get_last_fusion_stats()
            if isinstance(stats, dict):
                lr_stats = {
                    f"lr_{key}": value.detach() if torch.is_tensor(value) else value
                    for key, value in stats.items()
                }

        logits_hr_up = self._match_logits(logits_hr, masks.shape[-2:])
        logits_lr_up = self._match_logits(logits_lr, masks.shape[-2:])

        ce_hr = self.ce(logits_hr_up, masks) * self.w_ce_hr
        ce_lr = self.ce(logits_lr_up, masks) * self.w_ce_lr
        lapc_out = self._lapc_loss(dec_hr, dec_lr, logits_lr_up, masks)
        lapc = lapc_out["lapc"] * self.w_lapc
        total = ce_hr + ce_lr + lapc

        hr_miou, hr_pa = self._seg_metrics(logits_hr_up, masks)
        lr_miou, lr_pa = self._seg_metrics(logits_lr_up, masks)

        out: Dict[str, Any] = {
            "total": total,
            "ce_hr": ce_hr.detach(),
            "ce_lr": ce_lr.detach(),
            "lapc": lapc.detach(),
            "lapc_raw": lapc_out["lapc"].detach(),
            "lapc_class_count": lapc_out["lapc_class_count"].detach(),
            "lapc_conf_mean": lapc_out["lapc_conf_mean"].detach(),
            "lapc_proto_cos": lapc_out["lapc_proto_cos"].detach(),
            "hr_mIoU": hr_miou.detach(),
            "hr_pixel_acc": hr_pa.detach(),
            "lr_mIoU": lr_miou.detach(),
            "lr_pixel_acc": lr_pa.detach(),
            "lr_depth_keep_actual": zero_depth.new_tensor(0.0),
            "s_logits": logits_lr_up.detach(),
            "t_logits": logits_hr_up.detach(),
            "student_input": x_lr.detach(),
            "teacher_input": x_hr.detach(),
        }
        out.update(hr_stats)
        out.update(lr_stats)
        return out
