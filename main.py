import argparse
from pathlib import Path

import torch

from noisy_labels.model_config import ModelConfig
from noisy_labels.trainer import ModelTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser("Noisy Labels")
    parser.add_argument(
        "--test_path", help="Path to the corresponding test.json.gz", required=True
    )
    parser.add_argument(
        "--train_path",
        help="Path to the corresponding train.json.gz (optional)",
        required=False,
    )
    args = parser.parse_args()
    test_path = Path(args.test_path)
    if args.train_path:
        train_path = Path(args.train_path)
        if train_path.parent != test_path.parent:
            raise ValueError(
                f"Train path and Test path must be relative to the same dataset, found train {train_path.parent} and test {test_path.parent} "
            )
        for loss_type in [
            "cross_entropy",
            "ncod",
            "noisy_cross_entropy",
            "symmetric_cross_entropy",
            "symmetric_cross_entropy_weighted",
            "outlier_discounting_loss",
        ]:
            trainer = ModelTrainer(
                ModelConfig(train_path.parent.name),
                loss_type=loss_type,
                epochs=1,
                cycles=1,
            )
            trainer.train()


if __name__ == "__main__":
    main()
