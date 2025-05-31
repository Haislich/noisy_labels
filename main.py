import argparse
from pathlib import Path
from pprint import pprint

import torch
from loguru import logger

from noisy_labels.model_config import ModelConfig
from noisy_labels.trainer import ModelTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1
CYCLES = 3
ROUNDS = 1


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
        trainers = [
            ModelTrainer(
                ModelConfig(train_path.parent.name),
                loss_type=loss_type,
                epochs=EPOCHS,
                cycles=CYCLES,
            )
            for loss_type in [
                "cross_entropy_loss",
                # "ncod_loss",
                # "noisy_cross_entropy_loss",
                # "symmetric_cross_entropy_loss",
                # "weighted_symmetric_cross_entropy_loss",
                # "outlier_discounting_loss",
            ]
        ]
        results = {}
        winner_paths = None
        for round in range(ROUNDS):
            results[f"round_{round}"] = {}
            for trainer in trainers:
                results[f"round_{round}"][trainer.loss_type] = trainer.train(
                    winner_paths
                )

            winner_paths = []
            for loss in results["round_0"]:
                winner_paths.append(
                    max(results["round_0"][loss], key=lambda x: x["val_f1"])[
                        "model_path"
                    ],
                )


if __name__ == "__main__":
    main()
