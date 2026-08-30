from typing import List, Optional

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
    patch_size: List[int] = (128, 128, 128)
    in_channels: int = 4
    num_classes: int = 3
    base_num_features: int = 32
    max_num_features: int = 320
    num_downsampling: int = 5
    norm_type: str = "instance"          # "instance" | "batch" -> BN ablation
    deep_supervision: bool = True
    region_based_training: bool = False  # -> R ablation
    batch_dice: bool = False             # -> BD ablation
    dropout: float = 0.1
    pretrained_path: Optional[str] = None


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model", name="nnunet3d_model_schema", node=Nnunet3DModelSchema)