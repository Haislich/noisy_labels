import argparse
from pathlib import Path

import torch

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
            


if __name__ == "__main__":
    main()
