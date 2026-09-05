# """3D U-Net matching nnU-Net's generated architecture for BraTS (Fig. 1,
# Section 2.2 of Isensee et al., arXiv:2011.00848).
#
# Encoder/decoder with skip connections. Downsampling via strided
# convolutions (stride 2 on the first conv of each encoder stage after the
# first). Upsampling via ConvTranspose3d. Deep supervision heads branch off
# at all but the two lowest resolutions in the decoder (Fig. 1 caption).
# InstanceNorm3d/BatchNorm3d switch (BN ablation, Section 2.4). Softmax/
# sigmoid output switch (region-based training ablation, Section 2.3).
# """
#
# from collections.abc import Sequence
#
# import torch
# from torch import nn
#
#
# def _norm_layer(norm_type: str, num_channels: int) -> nn.Module:
#     if norm_type == "instance":
#         return nn.InstanceNorm3d(num_channels, affine=True)
#     if norm_type == "batch":
#         return nn.BatchNorm3d(num_channels)
#     raise ValueError(f"norm_type must be 'instance' or 'batch', got {norm_type!r}")
#
#
# class ConvNormLReLU(nn.Module):
#     """3x3x3 conv - Norm - LeakyReLU, per Fig. 1's basic block."""
#
#     def __init__(self, in_ch: int, out_ch: int, stride: int, norm_type: str):
#         super().__init__()
#         self.conv = nn.Conv3d(
#             in_ch,
#             out_ch,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=(norm_type != "batch"),
#         )
#         self.norm = _norm_layer(norm_type, out_ch)
#         self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.act(self.norm(self.conv(x)))
#
#
# class StackedConvBlock(nn.Module):
#     """Two ConvNormLReLU blocks. `stride` (2 for downsampling, 1 otherwise)
#     is applied on the first conv only, as nnU-Net performs downsampling via
#     strided convolutions rather than pooling."""
#
#     def __init__(self, in_ch: int, out_ch: int, stride: int, norm_type: str):
#         super().__init__()
#         self.block = nn.Sequential(
#             ConvNormLReLU(in_ch, out_ch, stride=stride, norm_type=norm_type),
#             ConvNormLReLU(out_ch, out_ch, stride=1, norm_type=norm_type),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)
#
#
# class SegmentationHead(nn.Module):
#     """1x1x1 conv - softmax/sigmoid, per Fig. 1's output block. Sigmoid is
#     used for region-based training (Section 2.3), softmax for the standard
#     3-class (edema/necrosis/enhancing) formulation."""
#
#     def __init__(self, in_ch: int, num_classes: int, region_based: bool):
#         super().__init__()
#         self.conv = nn.Conv3d(in_ch, num_classes, kernel_size=1)
#         self.region_based = region_based
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         logits = self.conv(x)
#         return torch.sigmoid(logits) if self.region_based else torch.softmax(logits, dim=1)
#
#
# class NNUNet3D(nn.Module):
#     """Full encoder/decoder. Constructor args mirror
#     `Nnunet3DModelSchema` field-for-field so it can be instantiated via
#     Hydra's `_target_` instantiation."""
#
#     def __init__(
#         self,
#         patch_size: Sequence[int] = (128, 128, 128),
#         in_channels: int = 4,
#         num_classes: int = 4,  # background + 3 foreground classes (NCR/ED/ET) — see model_schema.py's comment
#         base_num_features: int = 32,
#         max_num_features: int = 320,
#         num_downsampling: int = 5,
#         norm_type: str = "instance",
#         deep_supervision: bool = True,
#         region_based_training: bool = False,
#         batch_dice: bool = False,  # consumed by the loss, not the architecture; accepted for schema parity
#         dropout: float = 0.1,
#         pretrained_path: str | None = None,
#         **_ignored,  # tolerate extra schema fields (architecture, name, _target_, ...) from Hydra instantiation
#     ):
#         super().__init__()
#
#         # --- guard: bottleneck must not collapse below a sane spatial size ---
#         # Hit as a real bug during testing: with too many downsampling ops
#         # relative to patch_size, a spatial dim reaches 1 (or isn't evenly
#         # divisible), which breaks the stride-2 conv/InstanceNorm3d chain.
#         min_bottleneck_dim = 4
#         for dim in patch_size:
#             if dim % (2**num_downsampling) != 0:
#                 raise ValueError(
#                     f"patch_size dim {dim} is not evenly divisible by "
#                     f"2**num_downsampling ({2 ** num_downsampling}); "
#                     f"choose a patch_size that is a multiple of {2 ** num_downsampling}."
#                 )
#             bottleneck_dim = dim // (2**num_downsampling)
#             if bottleneck_dim < min_bottleneck_dim:
#                 raise ValueError(
#                     f"patch_size {tuple(patch_size)} with num_downsampling="
#                     f"{num_downsampling} collapses to a bottleneck spatial "
#                     f"size of {bottleneck_dim} (< {min_bottleneck_dim}); "
#                     f"reduce num_downsampling or increase patch_size."
#                 )
#
#         self.num_downsampling = num_downsampling
#         self.deep_supervision = deep_supervision
#         self.region_based_training = region_based_training
#         self.dropout_p = dropout
#
#         # channels per resolution level: 32, 64, 128, 256, 320, 320, ... (capped)
#         num_stages = num_downsampling + 1
#         self.features = [min(base_num_features * (2**i), max_num_features) for i in range(num_stages)]
#
#         # --- encoder ---
#         self.encoder_stages = nn.ModuleList()
#         in_ch = in_channels
#         for i, out_ch in enumerate(self.features):
#             stride = 1 if i == 0 else 2
#             self.encoder_stages.append(StackedConvBlock(in_ch, out_ch, stride=stride, norm_type=norm_type))
#             in_ch = out_ch
#         self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
#
#         # --- decoder ---
#         # decoder_stages[0] is the lowest-resolution upsample (out of the
#         # bottleneck), decoder_stages[-1] is full input resolution.
#         self.upsamples = nn.ModuleList()
#         self.decoder_stages = nn.ModuleList()
#         decoder_in = self.features[-1]
#         for i in range(num_downsampling - 1, -1, -1):
#             skip_ch = self.features[i]
#             self.upsamples.append(nn.ConvTranspose3d(decoder_in, skip_ch, kernel_size=2, stride=2))
#             self.decoder_stages.append(StackedConvBlock(skip_ch * 2, skip_ch, stride=1, norm_type=norm_type))
#             decoder_in = skip_ch
#
#         # --- deep supervision heads ---
#         # decoder_stages[i] (0-based, i=0 is the lowest-resolution upsample
#         # out of the bottleneck) produces `features[num_downsampling-1-i]`
#         # channels. Branch off at all but the two lowest decoder
#         # resolutions (Fig. 1 caption) -> skip i=0,1. The last index
#         # (i=num_downsampling-1) is always included here and is the
#         # full-resolution main output.
#         self.heads = nn.ModuleDict()
#         for i in range(2, num_downsampling):
#             self.heads[str(i)] = SegmentationHead(
#                 self.features[num_downsampling - 1 - i],
#                 num_classes,
#                 region_based_training,
#             )
#
#     def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
#         """Returns a list of segmentation outputs, ordered from lowest to
#         highest resolution among the active deep-supervision heads, with
#         the last element always the full-resolution main output. If
#         `deep_supervision` is False, returns a single-element list."""
#         skips = []
#         for stage in self.encoder_stages:
#             x = stage(x)
#             skips.append(x)
#         x = self.dropout(x)
#
#         outputs = []
#         for i, (up, dec) in enumerate(zip(self.upsamples, self.decoder_stages)):
#             skip_idx = self.num_downsampling - 1 - i
#             x = up(x)
#             x = torch.cat([x, skips[skip_idx]], dim=1)
#             x = dec(x)
#             head_key = str(i)
#             if head_key in self.heads and (self.deep_supervision or i == len(self.decoder_stages) - 1):
#                 outputs.append(self.heads[head_key](x))
#
#         return outputs
#
#
# def _count_params(model: nn.Module) -> int:
#     return sum(p.numel() for p in model.parameters())
#
#
# if __name__ == "__main__":
#     model = NNUNet3D()
#     x = torch.randn(1, 4, 128, 128, 128)
#     outs = model(x)
#     print(f"num deep-supervision outputs: {len(outs)}")
#     for o in outs:
#         print("  output shape:", tuple(o.shape))
#     print(f"total params: {_count_params(model):,} ({_count_params(model) / 1e6:.1f}M)")

"""3D U-Net matching nnU-Net's generated architecture for BraTS (Fig. 1,
Section 2.2 of Isensee et al., arXiv:2011.00848).

Encoder/decoder with skip connections. Downsampling via strided
convolutions (stride 2 on the first conv of each encoder stage after the
first). Upsampling via ConvTranspose3d. Deep supervision heads branch off
at all but the two lowest resolutions in the decoder (Fig. 1 caption).
InstanceNorm3d/BatchNorm3d switch (BN ablation, Section 2.4). Softmax/
sigmoid output switch (region-based training ablation, Section 2.3).
"""

from collections.abc import Sequence

import torch
from torch import nn


def _norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    raise ValueError(f"norm_type must be 'instance' or 'batch', got {norm_type!r}")


class ConvNormLReLU(nn.Module):
    """3x3x3 conv - Norm - LeakyReLU, per Fig. 1's basic block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, norm_type: str):
        super().__init__()
        self.conv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=(norm_type != "batch"),
        )
        self.norm = _norm_layer(norm_type, out_ch)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class StackedConvBlock(nn.Module):
    """Two ConvNormLReLU blocks. `stride` (2 for downsampling, 1 otherwise)
    is applied on the first conv only, as nnU-Net performs downsampling via
    strided convolutions rather than pooling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, norm_type: str):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormLReLU(in_ch, out_ch, stride=stride, norm_type=norm_type),
            ConvNormLReLU(out_ch, out_ch, stride=1, norm_type=norm_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationHead(nn.Module):
    """1x1x1 conv - softmax/sigmoid, per Fig. 1's output block. Sigmoid is
    used for region-based training (Section 2.3), softmax for the standard
    3-class (edema/necrosis/enhancing) formulation."""

    def __init__(self, in_ch: int, num_classes: int, region_based: bool):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, num_classes, kernel_size=1)
        self.region_based = region_based

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        return torch.sigmoid(logits) if self.region_based else torch.softmax(logits, dim=1)


class NNUNet3D(nn.Module):
    """Full encoder/decoder. Constructor args mirror
    `Nnunet3DModelSchema` field-for-field so it can be instantiated via
    Hydra's `_target_` instantiation."""

    def __init__(
        self,
        patch_size: Sequence[int] = (128, 128, 128),
        in_channels: int = 4,
        num_classes: int = 4,  # background + 3 foreground classes (NCR/ED/ET) — see model_schema.py's comment
        base_num_features: int = 32,
        max_num_features: int = 320,
        num_downsampling: int = 5,
        norm_type: str = "instance",
        deep_supervision: bool = True,
        region_based_training: bool = False,
        batch_dice: bool = False,  # consumed by the loss, not the architecture; accepted for schema parity
        dropout: float = 0.1,
        pretrained_path: str | None = None,
        **_ignored,  # tolerate extra schema fields (architecture, name, _target_, ...) from Hydra instantiation
    ):
        super().__init__()

        # --- guard: bottleneck must not collapse below a sane spatial size ---
        # Hit as a real bug during testing: with too many downsampling ops
        # relative to patch_size, a spatial dim reaches 1 (or isn't evenly
        # divisible), which breaks the stride-2 conv/InstanceNorm3d chain.
        min_bottleneck_dim = 4
        for dim in patch_size:
            if dim % (2**num_downsampling) != 0:
                raise ValueError(
                    f"patch_size dim {dim} is not evenly divisible by "
                    f"2**num_downsampling ({2 ** num_downsampling}); "
                    f"choose a patch_size that is a multiple of {2 ** num_downsampling}."
                )
            bottleneck_dim = dim // (2**num_downsampling)
            if bottleneck_dim < min_bottleneck_dim:
                raise ValueError(
                    f"patch_size {tuple(patch_size)} with num_downsampling="
                    f"{num_downsampling} collapses to a bottleneck spatial "
                    f"size of {bottleneck_dim} (< {min_bottleneck_dim}); "
                    f"reduce num_downsampling or increase patch_size."
                )

        # --- guard: at least one segmentation head must exist ---
        # Heads are built for decoder indices range(2, num_downsampling)
        # (Fig. 1: "all but the two lowest resolutions"). With
        # num_downsampling < 3, that range is empty — forward() would
        # return an empty list, and the actual failure would only surface
        # much later and far more confusingly (an IndexError deep inside
        # whatever loss function first tries to read preds[0]), not here
        # where the real problem is. Confirmed as a real failure mode via
        # an actual Phase 8 CV run using num_downsampling=2 for a fast
        # CPU smoke test.
        if num_downsampling < 3:
            raise ValueError(
                f"num_downsampling={num_downsampling} produces zero deep-supervision "
                f"heads (heads exist for decoder indices range(2, num_downsampling), "
                f"which is empty below 3); use num_downsampling >= 3."
            )

        self.num_downsampling = num_downsampling
        self.deep_supervision = deep_supervision
        self.region_based_training = region_based_training
        self.dropout_p = dropout

        # channels per resolution level: 32, 64, 128, 256, 320, 320, ... (capped)
        num_stages = num_downsampling + 1
        self.features = [min(base_num_features * (2**i), max_num_features) for i in range(num_stages)]

        # --- encoder ---
        self.encoder_stages = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(self.features):
            stride = 1 if i == 0 else 2
            self.encoder_stages.append(StackedConvBlock(in_ch, out_ch, stride=stride, norm_type=norm_type))
            in_ch = out_ch
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

        # --- decoder ---
        # decoder_stages[0] is the lowest-resolution upsample (out of the
        # bottleneck), decoder_stages[-1] is full input resolution.
        self.upsamples = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        decoder_in = self.features[-1]
        for i in range(num_downsampling - 1, -1, -1):
            skip_ch = self.features[i]
            self.upsamples.append(nn.ConvTranspose3d(decoder_in, skip_ch, kernel_size=2, stride=2))
            self.decoder_stages.append(StackedConvBlock(skip_ch * 2, skip_ch, stride=1, norm_type=norm_type))
            decoder_in = skip_ch

        # --- deep supervision heads ---
        # decoder_stages[i] (0-based, i=0 is the lowest-resolution upsample
        # out of the bottleneck) produces `features[num_downsampling-1-i]`
        # channels. Branch off at all but the two lowest decoder
        # resolutions (Fig. 1 caption) -> skip i=0,1. The last index
        # (i=num_downsampling-1) is always included here and is the
        # full-resolution main output.
        self.heads = nn.ModuleDict()
        for i in range(2, num_downsampling):
            self.heads[str(i)] = SegmentationHead(
                self.features[num_downsampling - 1 - i],
                num_classes,
                region_based_training,
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns a list of segmentation outputs, ordered from lowest to
        highest resolution among the active deep-supervision heads, with
        the last element always the full-resolution main output. If
        `deep_supervision` is False, returns a single-element list."""
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)
        x = self.dropout(x)

        outputs = []
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoder_stages)):
            skip_idx = self.num_downsampling - 1 - i
            x = up(x)
            x = torch.cat([x, skips[skip_idx]], dim=1)
            x = dec(x)
            head_key = str(i)
            if head_key in self.heads and (self.deep_supervision or i == len(self.decoder_stages) - 1):
                outputs.append(self.heads[head_key](x))

        return outputs


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = NNUNet3D()
    x = torch.randn(1, 4, 128, 128, 128)
    outs = model(x)
    print(f"num deep-supervision outputs: {len(outs)}")
    for o in outs:
        print("  output shape:", tuple(o.shape))
    print(f"total params: {_count_params(model):,} ({_count_params(model) / 1e6:.1f}M)")
