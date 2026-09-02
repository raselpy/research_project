from dataclasses import field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class ModelConfig:
    _target_: str = MISSING
    name: str = MISSING


@dataclass
class Nnunet3DModelSchema(ModelConfig):
    _target_: str = "src.models.nnunet3d.NNUNet3D"
    name: str = "nnunet3d"
    architecture: str = "3dunet"
    patch_size: list[int] = field(default_factory=lambda: [128, 128, 128])
    in_channels: int = 4
    # num_classes: 4 for the default (standard softmax) path — background
    # (0) + 3 foreground classes (NCR, ED, ET), matching
    # src/datasets/prepare.py's remap_labels() contiguous {0,1,2,3} output
    # exactly. The R ablation (region_based_training=True) must override
    # this to 3 alongside it: sigmoid over the 3 overlapping regions
    # (WT/TC/ET) has no separate background channel. Confirmed as a real
    # bug via an actual training run: num_classes=3 here silently broke
    # nn.functional.one_hot()/nll_loss() with "Class values must be
    # smaller than num_classes" the moment real 4-class label data hit
    # the loss function — not caught by any earlier structural test since
    # those used random target values within whatever range was assumed.
    num_classes: int = 4
    base_num_features: int = 32
    max_num_features: int = 320
    num_downsampling: int = 5
    norm_type: str = "instance"  # "instance" | "batch" -> BN ablation
    deep_supervision: bool = True
    region_based_training: bool = False  # -> R ablation
    batch_dice: bool = False  # -> BD ablation
    dropout: float = 0.1
    pretrained_path: str | None = None


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model", name="nnunet3d_model_schema", node=Nnunet3DModelSchema)
