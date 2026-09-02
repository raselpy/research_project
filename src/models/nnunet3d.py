"""3D U-Net matching nnU-Net's generated architecture for BraTS (Fig. 1,
Section 2.2 of Isensee et al., arXiv:2011.00848).
"""
from typing import List

import torch
import torch.nn as nn


def _norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "instance":
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    raise ValueError(f"norm_type must be 'instance' or 'batch', got {norm_type!r}")


class ConvNormLReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, norm_type: str):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=(norm_type != "batch"))
        self.norm = _norm_layer(norm_type, out_ch)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class StackedConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, norm_type: str):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormLReLU(in_ch, out_ch, stride=stride, norm_type=norm_type),
            ConvNormLReLU(out_ch, out_ch, stride=1, norm_type=norm_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, region_based: bool):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, num_classes, kernel_size=1)
        self.region_based = region_based

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        return torch.sigmoid(logits) if self.region_based else torch.softmax(logits, dim=1)


class NNUNet3D(nn.Module):
    def __init__(
        self,
        patch_size: List[int] = (128, 128, 128),
        in_channels: int = 4,
        num_classes: int = 4,
        base_num_features: int = 32,
        max_num_features: int = 320,
        num_downsampling: int = 5,
        norm_type: str = "instance",
        deep_supervision: bool = True,
        region_based_training: bool = False,
        batch_dice: bool = False,  # consumed by the loss, not the architecture; accepted for schema parity
        dropout: float = 0.1,
        pretrained_path: str = None,
        **_ignored,  # tolerate extra schema fields from Hydra instantiation
    ):
        super().__init__()

        # guard: bottleneck must not collapse below a sane spatial size
        min_bottleneck_dim = 4
        for dim in patch_size:
            if dim % (2 ** num_downsampling) != 0:
                raise ValueError(
                    f"patch_size dim {dim} is not evenly divisible by "
                    f"2**num_downsampling ({2 ** num_downsampling}); "
                    f"choose a patch_size that is a multiple of {2 ** num_downsampling}."
                )
            bottleneck_dim = dim // (2 ** num_downsampling)
            if bottleneck_dim < min_bottleneck_dim:
                raise ValueError(
                    f"patch_size {tuple(patch_size)} with num_downsampling="
                    f"{num_downsampling} collapses to a bottleneck spatial "
                    f"size of {bottleneck_dim} (< {min_bottleneck_dim}); "
                    f"reduce num_downsampling or increase patch_size."
                )

        self.num_downsampling = num_downsampling
        self.deep_supervision = deep_supervision
        self.region_based_training = region_based_training
        self.dropout_p = dropout

        num_stages = num_downsampling + 1
        self.features = [min(base_num_features * (2 ** i), max_num_features) for i in range(num_stages)]

        # encoder
        self.encoder_stages = nn.ModuleList()
        in_ch = in_channels
        for i, out_ch in enumerate(self.features):
            stride = 1 if i == 0 else 2
            self.encoder_stages.append(StackedConvBlock(in_ch, out_ch, stride=stride, norm_type=norm_type))
            in_ch = out_ch
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

        # decoder
        self.upsamples = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        decoder_in = self.features[-1]
        for i in range(num_downsampling - 1, -1, -1):
            skip_ch = self.features[i]
            self.upsamples.append(nn.ConvTranspose3d(decoder_in, skip_ch, kernel_size=2, stride=2))
            self.decoder_stages.append(StackedConvBlock(skip_ch * 2, skip_ch, stride=1, norm_type=norm_type))
            decoder_in = skip_ch

        # deep supervision heads: decoder_stages[i] (0-based, i=0 is the
        # lowest-resolution upsample out of the bottleneck) produces
        # features[num_downsampling-1-i] channels. Branch off at all but
        # the two lowest decoder resolutions -> skip i=0,1.
        self.heads = nn.ModuleDict()
        for i in range(2, num_downsampling):
            self.heads[str(i)] = SegmentationHead(
                self.features[num_downsampling - 1 - i], num_classes, region_based_training
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
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
