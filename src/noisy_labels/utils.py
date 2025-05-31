import os
import pickle
import tarfile
from pathlib import Path

import pandas as pd
import torch

from noisy_labels import logger
from noisy_labels.load_data import GraphDataset
from noisy_labels.models import EnsembleEdgeVGAE


def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.

    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    logger.info(f"Folder '{folder_path}' has been compressed into '{output_file}'")


def save_predictions(predictions, test_path):
    script_dir = os.getcwd()
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))

    os.makedirs(submission_folder, exist_ok=True)

    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({"id": test_graph_ids, "pred": predictions})

    output_df.to_csv(output_csv_path, index=False)
    logger.info(f"Predictions saved to {output_csv_path}")


def compute_class_weights(dataset, num_classes):
    counts = [0] * num_classes
    for g in dataset:
        if g.y is not None:
            counts[g.y.item()] += 1
    total = sum(counts)
    weights = [total / (c + 1e-6) for c in counts]
    norm_weights = torch.tensor(weights)
    norm_weights = norm_weights / norm_weights.sum()
    return norm_weights


def create_submission():
    for dataset_name in ["A", "B", "C", "D"]:
        test_path = f"./datasets/{dataset_name}/test.json.gz"
        predictions, _ = EnsembleEdgeVGAE(dataset_name).predict_with_ensemble_score(
            dataset_name
        )
        save_predictions(predictions, test_path)

    folder_path = Path("./submission/")  # Path to the folder you want to compress
    output_file = Path("./submission/submission.gz")  # Output .gz file name
    gzip_folder(folder_path, output_file)


def merge_pickles(paths, output_path):
    merged = []
    for path in paths:
        logger.info(f"Loading {path}...")
        with open(path, "rb") as f:
            data_list = pickle.load(f)  # Each file should be a list[IndexedData]
        merged.extend(data_list)

    logger.info(f"Saving merged data to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)
    logger.info("Done.")


def cacheABCD():
    # create the caches
    GraphDataset("./datasets/A/train.json.gz")
    GraphDataset("./datasets/B/train.json.gz")
    GraphDataset("./datasets/C/train.json.gz")
    GraphDataset("./datasets/D/train.json.gz")
    merge_pickles(
        [
            "./datasets/A/train.json.pkl",
            "./datasets/B/train.json.pkl",
            "./datasets/C/train.json.pkl",
            "./datasets/D/train.json.pkl",
        ],
        "./datasets/ABCD/train.json.pkl",
    )
