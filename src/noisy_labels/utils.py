import os
import random
import tarfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch


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


def create_submission():
    # for dataset_name in ["A","B","C","D"]:
    # Example usage
    folder_path = Path("./submission/")  # Path to the folder you want to compress
    output_file = Path("./submission/submission.gz")  # Output .gz file name
    gzip_folder(folder_path, output_file)


p = Path(os.getcwd()) / "checkpoints"
for checkpoint in p.glob("*.pth"):
    dataset_name = p / str(checkpoint.stem).split("_")[1]
    dataset_name.mkdir(exist_ok=True)
    dst = (dataset_name / checkpoint.stem.rsplit("_", 4)[0]).with_suffix(
        checkpoint.suffix
    )
    os.rename(checkpoint, dst)
    # print(checkpoint)
    # print(dataset_name)
    # print(dst)

    # break
