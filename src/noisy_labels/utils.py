import os
import tarfile
from pathlib import Path

import pandas as pd

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
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")


def save_predictions(predictions, test_path):
    script_dir = os.getcwd()
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))

    os.makedirs(submission_folder, exist_ok=True)

    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({"id": test_graph_ids, "pred": predictions})

    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def create_submission():
    for dataset_name in ["A", "B", "C", "D"]:
        model_paths = list(
            [
                Path(checkpoint)
                for checkpoint in Path(f"./checkpoints/{dataset_name}").glob(
                    "model*.pth"
                )
            ]
        )
        test_path = f"./datasets/{dataset_name}/test.json.gz"
        predictions, _ = EnsembleEdgeVGAE(model_paths).predict_with_ensemble_score(
            test_path
        )
        save_predictions(predictions, test_path)

    folder_path = Path("./submission/")  # Path to the folder you want to compress
    output_file = Path("./submission/submission.gz")  # Output .gz file name
    gzip_folder(folder_path, output_file)


create_submission()
