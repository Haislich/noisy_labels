
# Acknowledgments

This repository builds upon ideas and code from the [winning solution](https://sites.google.com/view/learning-with-noisy-graph-labe/winners) of the [IJCNN 2025 Competition: Learning with Noisy Graph Labels](https://sites.google.com/view/learning-with-noisy-graph-labe?usp=sharing), adapted for the Deep Learning Hackathon.

The original approach employs a **Variational Graph Autoencoder (VGAE)** to filter out noisy samples, an **ensemble of models** to address different noise conditions, and an improved **weighted voting mechanism** to enhance prediction accuracy.

We also make use of pretrained weights provided by the original authors. These weights follow the naming convention:

`model_[datasetname]_cycle_[cycle]_epoch_[epoch].pth`

We acknowledge and thank the authors of the original work for their valuable contribution of both code and pretrained models. You can find the original repository here: [Original Repo.](https://github.com/cminuttim/Learning-with-Noisy-Graph-Labels-Competition-IJCNN_2025)

## Extended Training Procedure with Loss Function Tournament and Intelligent Ensemble

![Image teaser](./images/teaser.svg)
This project improves the traditional training pipeline for learning from noisy labels through two key innovations:

1. **Weak pretraining using a loss function tournament**
2. **Final prediction via ensemble of best models**

## 1. Weak Pretraining with Loss Function Tournament

When the script is launched with both a `train_path` and a `test_path`, it checks for an aggregated dataset named `ABCD/train.json.gz`. If found, each available loss function competes in a tournament.

Each participant (i.e., loss function) trains a model on the ABCD dataset for a set number of **rounds**, each made of **cycles** and **epochs**. After each round:

- The best-performing model for each loss is selected.
- Then, the loss function with the highest F1 score overall is chosen for finetuning.

## 2. Finetuning and Pre-trained Model Management

Once weak pretraining is completed (or skipped if no ABCD data is found), finetuning is performed on the provided dataset.

During finetuning:

- Previously trained models are loaded from disk (if available).
- The worst-performing model (based on validation F1 score) is selected.
- A new model is trained using the best-performing loss from the pretraining phase.
- If the new model outperforms the worst in the pool, it **replaces it**—both on disk and in the internal metadata.

This guarantees a continuously improving collection of checkpoints while limiting their total number.

## 3. Best Models Ensemble

After training, an ensemble is built over the saved models to generate final predictions on the test set. This is done as follows:

1. Load all available trained models.
2. Sort them by validation F1 score.
3. Select the top-k models (default behavior assumed to be top 5).
4. Aggregate predictions using ensemble voting or averaging.

This method avoids blindly trusting one "best" model per loss and instead builds a more robust consensus prediction.
