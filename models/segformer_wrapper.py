# models/segformer_wrapper.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from config import DATA

_SOURCES = {
    "segformerb0": "nvidia/mit-b0",
    "segformerb3": "nvidia/mit-b3",
}


class SegFormerWrapper(nn.Module):
    """
    ImageNet 으로 사전학습된 SegFormer encoder 를 불러오는 wrapper.

        forward(x) -> logits (upsampled to input size)
        forward(x, return_feats=True) -> (logits, feats)
        forward(x, return_embeds=True) -> (logits, embeds)
        forward(x, return_feats=True, return_embeds=True) -> (logits, feats, embeds)
        forward(x, is_feat=True) -> (feats, logits, embeds, dec_fuse)

    feats 는 encoder stage 출력, dec_fuse 는 All-MLP decoder 의 fused feature 로
    각각 EVWF-KD 의 encoder / decoder distillation 에 쓰인다.
    """

    def __init__(self, name: str, num_classes: int = DATA.NUM_CLASSES,
                 ignore_index: int | None = None):
        super().__init__()
        name = name.lower()
        assert name in _SOURCES, f"Unknown SegFormer name: {name}"
        src = _SOURCES[name]

        # id2label/label2id는 num_labels를 덮어쓴다(HF가 len(id2label)로 재계산).
        # 따라서 DATA.CLASS_NAMES를 그대로 넣으면 num_classes 인자가 무시되므로,
        # 길이가 맞을 때만 이름을 쓰고 아니면 인덱스로 채운다.
        if len(DATA.CLASS_NAMES) == num_classes:
            labels = list(DATA.CLASS_NAMES)
        else:
            labels = [f"class_{i}" for i in range(num_classes)]

        cfg = SegformerConfig.from_pretrained(src)
        cfg.id2label = {i: n for i, n in enumerate(labels)}
        cfg.label2id = {n: i for i, n in enumerate(labels)}
        cfg.num_labels = num_classes
        cfg.output_hidden_states = True
        cfg.return_dict = True
        # void ignore index (CamVid: 11, MiniCity: 19)
        if ignore_index is None:
            ignore_index = DATA.IGNORE_INDEX if len(DATA.CLASS_NAMES) == num_classes else num_classes - 1
        cfg.semantic_loss_ignore_index = int(ignore_index)

        assert cfg.num_labels == num_classes, (
            f"num_labels={cfg.num_labels} != requested {num_classes}"
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            src, config=cfg, ignore_mismatched_sizes=True
        )

        self._last_encoder_feats = None   # Tuple[Tensor(B,C,H,W), ...] len 4
        self._last_encoder_embeds = None  # Tuple[Tensor(B,N,C), ...]   len 4

        # ---- decoder fuse feature cache (decode_head.linear_fuse output) ----
        self._last_decoder_fuse = None  # Tensor(B, C_dec, H/4, W/4) typically
        self._dec_hooks = []

        # ---- patch embedding hook cache ----
        self._patch_embed_cache = [None, None, None, None]     # (B,N,C)
        self._patch_hw_cache = [None, None, None, None]        # (H,W) if obtainable
        self._patch_hooks = []
        self._register_patch_embedding_hooks()
        self.force_patch_embeds = False

        self._register_decoder_fuse_hook()

    def set_force_patch_embeds(self, flag: bool = True):
        self.force_patch_embeds = bool(flag)

    # --------- 내부 유틸 ---------
    def _register_patch_embedding_hooks(self):
        # remove old
        for h in self._patch_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._patch_hooks = []

        enc = getattr(getattr(self.model, "segformer", None), "encoder", None)
        if enc is None:
            return
        pel = getattr(enc, "patch_embeddings", None)
        if pel is None or not hasattr(pel, "__len__"):
            return

        def _make_hook(i: int):
            def _hook(module, inp, out):
                # HF OverlapPatchEmbeddings forward usually returns (embeddings, height, width)
                if isinstance(out, (tuple, list)):
                    emb = out[0]
                    h = out[1] if len(out) > 1 else None
                    w = out[2] if len(out) > 2 else None
                else:
                    emb, h, w = out, None, None

                if torch.is_tensor(emb):
                    self._patch_embed_cache[i] = emb  # (B,N,C)
                if isinstance(h, int) and isinstance(w, int):
                    self._patch_hw_cache[i] = (h, w)
            return _hook

        n = min(4, len(pel))
        for i in range(n):
            self._patch_hooks.append(pel[i].register_forward_hook(_make_hook(i)))

    def _register_decoder_fuse_hook(self):
        """
        SegFormer decode head의 fusion feature를 얻기 위한 hook.
        - HuggingFace SegFormer는 decode_head.linear_fuse (Conv2d)에서
          concat된 multi-scale feature를 C_dec로 fuse한 텐서를 만든다.
        - 해당 텐서는 classifier(1x1 conv) 직전 feature로 간주 가능.
        """
        # remove old
        for h in self._dec_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._dec_hooks = []

        # HF: SegformerForSemanticSegmentation has attribute `decode_head`
        dec = getattr(self.model, "decode_head", None)
        if dec is None:
            return

        # Most stable target: decode_head.linear_fuse (Conv2d)
        fuse = getattr(dec, "linear_fuse", None)
        if fuse is None:
            # fallback: if no linear_fuse, hook decode_head output itself (less guaranteed)
            def _dec_hook(module, inp, out):
                if torch.is_tensor(out):
                    self._last_decoder_fuse = out
                elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    self._last_decoder_fuse = out[0]
            self._dec_hooks.append(dec.register_forward_hook(_dec_hook))
            return

        def _fuse_hook(module, inp, out):
            # out should be Tensor(B, C_dec, H/4, W/4)
            if torch.is_tensor(out):
                self._last_decoder_fuse = out

        self._dec_hooks.append(fuse.register_forward_hook(_fuse_hook))

    def get_last_decoder_fuse(self):
        return self._last_decoder_fuse

    def _reshape_hidden_state(self, feat: torch.Tensor, idx: int, hw):
        """
        SegFormer encoder hidden state (B, HW, C) → (B, C, H, W)
        - feat: (B, HW, C) or (B, C, H, W)
        - idx : stage index (0~3) within LAST-4 stages
        - hw  : input image (H_in, W_in)
        """
        if feat.dim() == 4:
            return feat
        if feat.dim() != 3:
            raise RuntimeError(f"SegFormer hidden state must be 3D/4D, got {tuple(feat.shape)}")

        b, n, c = feat.shape
        h_in, w_in = hw

        ratios = getattr(self.model.config, "downsample_ratios", None)
        strides = getattr(self.model.config, "encoder_stride", None)
        if ratios and idx < len(ratios):
            stride = ratios[idx]
        elif strides and idx < len(strides):
            stride = strides[idx]
        else:
            stride = 2 ** (idx + 2)  # fallback

        h = max(1, math.ceil(h_in / stride))
        w = max(1, math.ceil(w_in / stride))
        if h * w != n:
            # fallback: make near-square
            h = max(1, int(round(math.sqrt(n))))
            w = max(1, n // h)
            if h * w != n:
                raise RuntimeError(f"Cannot reshape hidden state length {n} to 2D (stage {idx})")

        feat = feat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return feat

    def _collect_encoder_feats_and_embeds(self, out, input_hw):
        """
        - feats : stage 4개 feature map (B,C,H,W)
        - embeds: stage 4개 patch embedding (B,N,C)  (hook 우선, 실패 시 hidden_states 기반)
        """
        feats = getattr(out, "encoder_hidden_states", None)
        if feats is None:
            feats = getattr(out, "hidden_states", None)
        if feats is None:
            self._last_encoder_feats = None
            self._last_encoder_embeds = None
            return None, None

        feats = feats[-4:]  # last 4 stages

        # ---- embeds (hook-based first) ----
        hook_ok = all(self._patch_embed_cache[i] is not None for i in range(4))
        if self.force_patch_embeds and not hook_ok:
            raise RuntimeError(
                "Stage distillation requires patch embeddings from the "
                "encoder.patch_embeddings hooks, but the hook cache is empty "
                "(transformers version mismatch?)."
            )

        if hook_ok:
            embeds = tuple(self._patch_embed_cache[i] for i in range(4))
        else:
            embeds = tuple(f if f.dim() == 3 else f.flatten(2).transpose(1, 2) for f in feats)

        # ---- feature maps ----
        reshaped = tuple(self._reshape_hidden_state(feat, idx, input_hw) for idx, feat in enumerate(feats))

        self._last_encoder_feats = reshaped
        self._last_encoder_embeds = embeds
        return reshaped, embeds

    def get_last_encoder_features(self):
        return self._last_encoder_feats

    def get_last_encoder_embeddings(self):
        return self._last_encoder_embeds

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
        return_embeds: bool = False,
        is_feat: bool = False,
    ):
        """
        - 기본: logits
        - 기존 호환: (logits, feats), (logits, embeds), (logits, feats, embeds)
        - is_feat=True -> (feats, logits, embeds, dec_fuse)
        """
        # clear caches each forward
        for i in range(4):
            self._patch_embed_cache[i] = None
            self._patch_hw_cache[i] = None
        self._last_decoder_fuse = None

        out = self.model(pixel_values=x, return_dict=True)

        logits = F.interpolate(
            out.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        feats, embeds = self._collect_encoder_feats_and_embeds(out, x.shape[-2:])

        # ---- feature distillation path ----
        if is_feat:
            if feats is None or embeds is None:
                raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
            # decoder fuse feature는 hook으로 채워짐 (없을 수도 있으니 None 허용)
            dec_fuse = self._last_decoder_fuse
            return feats, logits, embeds, dec_fuse

        # ---- 기존 호환 path ----
        if not return_feats and not return_embeds:
            return logits

        if return_feats and not return_embeds:
            if feats is None:
                raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
            return logits, feats

        if not return_feats and return_embeds:
            if embeds is None:
                raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
            return logits, embeds

        if feats is None or embeds is None:
            raise RuntimeError("SegFormer hidden_states를 얻지 못했습니다.")
        return logits, feats, embeds
