# source/trainer.py
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import random_split  # ,WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm.auto import tqdm

from noisy_labels.load_data import GraphDataset, IndexedData, IndexedSubset
from noisy_labels.loss import NCODLoss, NoisyCrossEntropyLoss, SymmetricCrossEntropyLoss
from noisy_labels.model_config import ModelConfig
from noisy_labels.models import EdgeVGAE


def warm_up_lr(epoch, num_epoch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params["lr"] = (epoch + 1) ** 3 * init_lr / num_epoch_warm_up**3


class ModelTrainer:
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.config = config
        self.device = device
        self.dataset = GraphDataset(
            Path(f"./datasets/{self.config.dataset_name}/train.json.gz"),
        )
        self.best_f1_scores = []
        self.eval_criterion = torch.nn.CrossEntropyLoss()

        self.checkpoints_dir = Path(f"./checkpoints/{self.config.dataset_name}")
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.pretrained_models = list(self.checkpoints_dir.glob("model*.pth"))
        # self.submissions_dir = Path(f"./submissions/{self.config.dataset_name}")
        # self.submissions_dir.mkdir(exist_ok=True)

        self.logs_dir = Path(f"./logs/{self.config.dataset_name}")
        self.logs_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.logs_dir / "training.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
        )

    def dataset_setup(self, seed: int):
        val_size = int(0.2 * len(self.dataset))
        train_size = len(self.dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            self.dataset, [train_size, val_size], generator=generator
        )
        train_dataset = IndexedSubset(train_dataset)
        val_dataset = IndexedSubset(val_dataset)
        return (train_dataset, val_dataset)

    def _eval_epoch(
        self,
        model: EdgeVGAE,
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                z, mu, logvar, class_logits = model(
                    data.x, data.edge_index, data.edge_attr, data.batch
                )

                # Calculate CrossEntropyLoss
                loss = self.eval_criterion(class_logits, data.y)

                # Get predictions
                pred_classes = class_logits.argmax(dim=1).cpu().numpy()
                true_labels = data.y.cpu().numpy()

                # Accumulate predictions and labels for F1 score
                all_preds.extend(pred_classes)
                all_labels.extend(true_labels)

                # Accumulate loss
                batch_size = data.y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = float(total_loss / total_samples)
        f1 = float(f1_score(all_labels, all_preds, average="weighted"))

        return {
            "cross_entropy_loss": avg_loss,
            "f1_score": f1,
            "num_samples": total_samples,
        }

    def _train_epoch(
        self,
        model: EdgeVGAE,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_u: Optional[torch.optim.Optimizer],
        epoch: int,
        scheduler,
    ):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for data in train_loader:
            data: IndexedData = data.to(self.device)
            optimizer.zero_grad()
            if optimizer_u is not None:
                optimizer_u.zero_grad()

            # Forward pass
            z, mu, logvar, class_logits = model(
                data.x, data.edge_index, data.edge_attr, data.batch
            )

            # Calculate losses
            recon_loss = model.recon_loss(z, data.edge_index, data.edge_attr)
            kl_loss = model.kl_loss(mu, logvar)
            if isinstance(criterion, NCODLoss):
                graph_embeddings = global_mean_pool(z, data.batch)
                class_loss = criterion(
                    logits=class_logits,
                    indexes=data.idx,
                    embeddings=graph_embeddings,
                    targets=data.y,
                    epoch=epoch,
                )
            else:
                class_loss = criterion(class_logits, data.y)

            # Total loss
            loss = 0.15 * recon_loss + 0.1 * kl_loss + class_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if optimizer_u is not None:
                optimizer_u.step()
            # scheduler.step()

            # Accumulate loss
            batch_size = data.y.size(0)  # type: ignore
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return float(total_loss / total_samples)

    def _train_single_cycle(
        self,
        cycle: int,
        criterion: nn.Module,
        train_data: IndexedSubset,
        val_data: IndexedSubset,
    ):
        logging.info(f"\nStarting training cycle {cycle}")
        print(f"\nStarting training cycle {cycle}")

        # Load pretrained models if any
        if len(self.pretrained_models) > 0:
            n = len(self.pretrained_models)
            model_path = self.pretrained_models[(cycle - 1) % n]
            model = EdgeVGAE.from_pretrained(model_path)
            logging.info(f"Loaded pretrained model: {model_path}")
            print(f"Loaded pretrained model: {model_path}")

        else:
            model = EdgeVGAE(
                self.config.input_dim,
                self.config.edge_dim,
                self.config.hidden_dim,
                self.config.latent_dim,
                self.config.num_classes,
            ).to(self.device)

        train_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=self.config.batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        optimizer_u = None
        if isinstance(criterion, NCODLoss):
            optimizer_u = torch.optim.SGD(criterion.parameters(), lr=1e-3)

        # Warm-up scheduler
        warmup_epochs = self.config.warmup  # Number of warm-up epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Monitor validation
            factor=0.7,  # Reduce LR by 50% on plateau
            patience=10,  # Number of epochs with no improvement
            min_lr=1e-6,
        )

        best_val_loss = float("inf")
        # Track best F1 score even though we're not using it for selection
        best_f1 = 0.0
        epoch_best = 0
        best_model_path = None
        progress_bar = tqdm(range(self.config.epochs), "")
        for epoch in progress_bar:
            progress_bar.set_description_str(f"Epoch {epoch + 1}/{self.config.epochs}")

            # Warm-up phase
            if epoch < warmup_epochs:
                warm_up_lr(epoch, warmup_epochs, self.config.learning_rate, optimizer)
                if epoch == (warmup_epochs - 1):
                    logging.info("Warm-up epochs finished")
                    print("Warm-up epochs finished")

            # Training
            model.train()
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, optimizer_u, epoch, scheduler
            )

            # Validation
            val_metrics = self._eval_epoch(model, val_loader)
            val_loss = val_metrics["cross_entropy_loss"]
            val_f1 = val_metrics["f1_score"]

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Cycle {cycle}, Epoch {epoch + 1}, LR {optimizer.param_groups[0]['lr']:.3e}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val F1: {val_f1:.4f}"
                )

            # Update lr after warm-up (Scheduler ReduceLROnPlateau)
            if epoch >= warmup_epochs:
                scheduler.step(val_f1)

            # Save model if it has the best f1 score so far
            if val_f1 > best_f1:
                best_val_loss = val_loss
                best_f1 = val_f1
                epoch_best = epoch

                # Save the model with both metrics in filename
                best_model_path = (
                    self.checkpoints_dir
                    / f"cycle_{cycle}_epoch_{epoch}_"
                    / f"loss_{val_loss}_f1_{val_f1}.pth"
                )
                # model.save(
                #     self.checkpoints_dir / f"cycle_{cycle}_epoch_{epoch}",
                #     val_loss,
                #     val_f1,
                #     train_loss,
                #     self.config,
                #     epoch,
                #     cycle,
                # )

                logging.info(f"New best model saved: {best_model_path}")
                logging.info(
                    f"Best validation metrics - Loss: {val_loss:.4f}, F1: {val_f1:.4f}"
                )
                print(f"New best model saved: {best_model_path}")
                print(
                    f"Best validation metrics - Loss: {val_loss:.4f}, F1: {val_f1:.4f}"
                )

            # If there is no improvement, reload the best parameters
            if (
                (epoch - epoch_best) > self.config.early_stopping_patience // 2
                and epoch % 10 == 0
                # Reload only if there is something to reload
                and best_model_path is not None
            ):
                model = EdgeVGAE.from_pretrained(best_model_path)
                logging.info(f"Reloading best model: {best_model_path}")
                print(f"Reloading best model: {best_model_path}")

            # Early stopping based on validation loss
            if (epoch - epoch_best) > self.config.early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                print(f"Early stopping triggered at epoch {epoch}")
                break
        progress_bar.close()
        # self.models.append(best_model_path)
        return best_val_loss, best_f1, best_model_path

    def train(
        self,
        loss_type: Literal[
            "cross_entropy", "ncod", "noisy_cross_entropy", "symmetric_cross_entropy"
        ],
        cycles=10,
    ):
        logging.info("Starting training")
        print("Starting training")

        results: List[Dict[str, int | float | Optional[Path]]] = []
        progress_bar = tqdm(range(cycles))
        if loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif loss_type == "ncod":
            criterion = NCODLoss(
                self.dataset, embedding_dimensions=self.config.latent_dim
            )

        elif loss_type == "noisy_cross_entropy":
            criterion = NoisyCrossEntropyLoss(p_noisy=0.2)

        elif loss_type == "symmetric_cross_entropy":
            criterion = SymmetricCrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss, {loss_type} is not a valid loss.")

        for cycle in progress_bar:
            progress_bar.set_description(f"Cycle {cycle}/{cycles}")
            cycle_seed = cycle + 1

            logging.info(f"\nStarting cycle {cycle + 1} with seed {cycle_seed}")

            train_data, val_data = self.dataset_setup(seed=cycle + 1)
            val_loss, val_f1, model_path = self._train_single_cycle(
                cycle + 1,
                criterion,
                train_data,
                val_data,
            )

            results.append(
                {
                    "cycle": cycle + 1,
                    "seed": cycle_seed,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "model_path": model_path,
                }
            )

            # Log summary of this cycle
            logging.info(f"Cycle {cycle + 1} completed:")
            logging.info(f"- Final validation loss: {val_loss:.4f}")
            logging.info(f"- Final F1 score: {val_f1:.4f}")
            print(f"Cycle {cycle + 1} completed:")
            print(f"- Final validation loss: {val_loss:.4f}")
            print(f"- Final F1 score: {val_f1:.4f}")
        return results


ModelTrainer(
    ModelConfig("A", epochs=1),
).train("ncod", cycles=1)
