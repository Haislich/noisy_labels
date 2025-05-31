from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    dataset_name: Literal["A", "B", "C", "D"] | str
    input_dim: int = 1
    edge_dim: int = 7
    batch_size: int = 64
    hidden_dim: int = 128
    latent_dim: int = 8
    num_classes: int = 6
