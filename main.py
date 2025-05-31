import argparse
import gc
import os
import random
from pathlib import Path

import torch
from loguru import logger

from noisy_labels.model_config import ModelConfig
from noisy_labels.models import EnsembleEdgeVGAE
from noisy_labels.trainer import ModelTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1
CYCLES = 1
ROUNDS = 1
ROOT = os.getcwd()
DEFAULT_LOSS = "cross_entropy_loss"
LOSS_TYPES = [
    "cross_entropy_loss",
    "ncod_loss",
    "noisy_cross_entropy_loss",
    "symmetric_cross_entropy_loss",
    "weighted_symmetric_cross_entropy_loss",
    "outlier_discounting_loss",
]


def losses_tournament(
    dataset_name: str,
    epochs: int,
    cycles: int,
    rounds: int,
    config_stage: str,
):
    trainers = [
        ModelTrainer(
            ModelConfig(dataset_name),
            loss_type=loss,
            epochs=epochs,
            cycles=cycles,
        )
        for loss in LOSS_TYPES
    ]
    results, best_losses, best_paths = {}, [], []

    for round_idx in range(rounds):
        logger.info(f"{config_stage} - Round {round_idx + 1}/{rounds}")
        results[f"round_{round_idx}"] = {}
        random.shuffle(trainers)

        for trainer in trainers:
            results[f"round_{round_idx}"][trainer.loss_type] = trainer.train()

        round_winners = {
            loss: max(entries, key=lambda x: x["val_f1"])
            for loss, entries in results[f"round_{round_idx}"].items()
        }
        best_loss = max(round_winners, key=lambda k: round_winners[k]["val_f1"])
        best_losses.append(best_loss)
        best_paths += [Path(entry["model_path"]) for entry in round_winners.values()]

        logger.info(
            "Best F1 scores this round:\n"
            + "\n".join(f"{k}: {v['val_f1']:.4f}" for k, v in round_winners.items())
        )
        logger.info(f"Best round loss: {best_loss}")

    # Clean up memory
    del trainers
    gc.collect()
    return best_losses, best_paths


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a model on noisy labeled datasets."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to test.json.gz (e.g., ./datasets/XYZ/test.json.gz)",
    )
    parser.add_argument(
        "--train_path", type=str, help="Optional path to train.json.gz for training"
    )
    args = parser.parse_args()

    test_path = Path(args.test_path)
    dataset_name = test_path.parent.name

    if args.train_path:
        train_path = Path(args.train_path)
        if train_path.parent != test_path.parent:
            raise ValueError(
                f"Train path and Test path must be in the same dataset folder.\n"
                f"Got: train={train_path.parent}, test={test_path.parent}"
            )

        checkpoint_dir = Path(f"./checkpoints/{dataset_name}")
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            logger.info(
                f"Starting weak pretraining on ABCD ({ROUNDS=}, {CYCLES=}, {EPOCHS=})"
            )
            losses_tournament(
                "ABCD", EPOCHS, CYCLES, ROUNDS, config_stage="Pretraining"
            )

        logger.info(f"Finetuning on {dataset_name}")
        best_losses, best_model_paths = losses_tournament(
            dataset_name, EPOCHS, CYCLES, ROUNDS, config_stage="Finetuning"
        )

        best_loss = max(set(best_losses), key=best_losses.count)
        trainer = ModelTrainer(
            ModelConfig(dataset_name), loss_type=best_loss, epochs=EPOCHS, cycles=CYCLES
        )
        trainer.train(best_model_paths)

    EnsembleEdgeVGAE(dataset_name).predict_with_ensemble_score(dataset_name)


if __name__ == "__main__":
    main()
