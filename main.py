import argparse
import gc
from pathlib import Path

import torch
from loguru import logger

from noisy_labels.model_config import ModelConfig
from noisy_labels.trainer import ModelTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1
CYCLES = 3
ROUNDS = 1
DEFAULT_LOSS = "cross_entropy_loss"


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
        #  /
        logger.info(
            f"Starting weak pretrain on ABCD, for {ROUNDS=}, {CYCLES=},{EPOCHS=}"
        )
        trainers = [
            ModelTrainer(
                ModelConfig("ABCD"),
                loss_type=loss_type,
                epochs=EPOCHS,
                cycles=CYCLES,
            )
            for loss_type in [
                "cross_entropy_loss",
                "ncod_loss",
                # "noisy_cross_entropy_loss",
                # "symmetric_cross_entropy_loss",
                # "weighted_symmetric_cross_entropy_loss",
                # "outlier_discounting_loss",
            ]
        ]
        results = {}
        pretrained_model_paths = None
        best_overall_loss = DEFAULT_LOSS
        for round in range(ROUNDS):
            results[f"round_{round}"] = {}
            for trainer in trainers:
                results[f"round_{round}"][trainer.loss_type] = trainer.train(
                    pretrained_model_paths
                )

            winner_paths = []
            for loss in results["round_0"]:
                winner_paths.append(
                    max(results["round_0"][loss], key=lambda x: x["val_f1"])[
                        "model_path"
                    ],
                )
            winners = {}
            for loss in results["round_0"].keys():
                winners[loss] = max(results["round_0"][loss], key=lambda x: x["val_f1"])
            pretrained_model_paths = [
                Path(elem["model_path"]) for elem in winners.values()
            ]
            best_overall_loss: str = max(winners, key=lambda x: winners[x]["val_f1"])
            logger.info(
                "Best F1 scores for each model: "
                + "\n".join((f"{k}:{v['val_f1']}" for k, v in winners.items()))
            )
            logger.info(f"Best overall loss {best_overall_loss}")

        logger.info(
            f"Finished weak pretraining of ABCD, starting training on {train_path.parent.name}"
        )
        for trainer in trainers:
            del trainer
        del trainers
        gc.collect()

        trainer = ModelTrainer(
            ModelConfig(train_path.parent.name),
            loss_type=best_overall_loss,
            epochs=EPOCHS,
            cycles=CYCLES,
        )
        trainer.train()


if __name__ == "__main__":
    main()
