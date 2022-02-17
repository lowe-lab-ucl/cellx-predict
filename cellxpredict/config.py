import enum
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class Models(enum.Enum):
    ENCODER = enum.auto()
    PROJECTOR = enum.auto()
    TEMPORAL = enum.auto()
    FULL = enum.auto()


@dataclass
class ConfigBase:
    model: str = ""
    src_dir: Optional[os.PathLike] = None
    log_dir: Optional[os.PathLike] = None
    docs_dir: Optional[os.PathLike] = None
    model_dir: Optional[os.PathLike] = None
    latent_dims: int = 32
    intermediate_dims: int = 256
    capacity: int = 50
    gamma: int = 1_000
    input_shape: Tuple[int, int, int] = (64, 64, 2)
    layers: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    def filename(self, component: str = "weights"):
        filename = (
            f"{self.model}_{component}_capacity={self.capacity}_"
            f"z={self.latent_dims}_gamma={self.gamma}"
        )
        return filename

    def filename_brief(self, component: str = "weights"):
        filename = (f"{self.model}_{component}")
        return filename


@dataclass
class EncoderConfig(ConfigBase):
    model: str = "encoder"
    epochs: int = 50
    batch_size: int = 256
    max_iterations: int = 100


@dataclass
class DecoderConfig(ConfigBase):
    model: str = "decoder"


@dataclass
class ProjectorConfig(ConfigBase):
    model: str = "projector"


@dataclass
class TemporalConfig(ConfigBase):
    model: str = "temporal"
    epochs: int = 100
    batch_size: int = 128
    num_outputs: int = 3
    max_len: int = 128
    dropout_rate: float = 0.3
    noise: float = 1.0
    use_probabilistic_encoder: bool = False
    use_rotations: bool = False


@dataclass
class FullConfig(ConfigBase):
    model: str = "full"
    num_outputs: int = 3
    max_len: int = 128
    dropout_rate: float = 0.0
    noise: float = 1.0
    use_probabilistic_encoder: bool = False


def config_from_args(args) -> ConfigBase:
    """Convert cmd line arguments to a config."""
    cfg_class = getattr(sys.modules[__name__], f"{args.model.title()}Config")
    cfg = cfg_class()
    args = vars(args)
    for arg, value in args.items():
        setattr(cfg, str(arg), value)
    return cfg


if __name__ == "__main__":
    pass
