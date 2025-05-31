from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    dataset_path: Path
    input_dim: int = 1
    edge_dim: int = 7
    batch_size: int = 64
    hidden_dim: int = 128
    latent_dim: int = 8
    num_classes: int = 6
