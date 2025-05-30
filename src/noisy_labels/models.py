# source/models.py - Contains all model definitions
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

from noisy_labels.model_config import ModelConfig


# Edge-aware encoder with classification capability
class EdgeEncoder(MessagePassing):
    def __init__(self, in_channels, edge_dim, hidden_dim):
        super(EdgeEncoder, self).__init__(aggr="add")  # Message aggregation
        self.node_mlp = torch.nn.Linear(in_channels + hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.LeakyReLU(0.15),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_mlp(edge_attr)  # Transform edge features
        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, edge_attr], dim=1)  # Concatenate node and edge features
        return self.node_mlp(z)

    def update(self, aggr_out):
        return aggr_out


class EdgeVGAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, latent_dim):
        super(EdgeVGAEEncoder, self).__init__()
        self.conv1 = EdgeEncoder(input_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeEncoder(hidden_dim, edge_dim, hidden_dim)
        self.drop = torch.nn.Dropout(0.05)

        # Mean and log variance layers
        self.mu_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.drop(x)
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr), 0.15)
        x = self.drop(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr), 0.15)
        # x = self.drop(x)
        return self.mu_layer(x), self.logvar_layer(x)  # Return mean and log variance


class EdgeVGAE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_classes: int,
    ):
        super(EdgeVGAE, self).__init__()
        self.encoder = EdgeVGAEEncoder(input_dim, edge_dim, hidden_dim, latent_dim)
        self.classifier = torch.nn.Linear(latent_dim, num_classes)  # Classifier head

        # MLP for edge attribute reconstruction
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, latent_dim),
            torch.nn.LeakyReLU(0.15),
            torch.nn.Linear(latent_dim, edge_dim),
        )

        # Initialize weights using Kaiming initialization
        self.init_weights()

    @staticmethod
    def from_pretrained(
        model_path: Path | str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        model_path_root = model_path.parent
        models_metadata_path = model_path_root / "metadata.json"
        with open(models_metadata_path, "r") as models_metadata_fp:
            models_metadata: Dict = json.load(models_metadata_fp)[model_path.stem]
            model_config_dict: Optional[Dict] = models_metadata.get("config", None)
            if model_config_dict is None:
                raise ValueError("No valid configuration found for model")
            else:
                model = EdgeVGAE(
                    model_config_dict["input_dim"],
                    model_config_dict["edge_dim"],
                    model_config_dict["hidden_dim"],
                    model_config_dict["latent_dim"],
                    model_config_dict["num_classes"],
                ).to(device)

            models_state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(models_state_dict)
            return model

    def save(
        self,
        model_path: Path | str,
        val_loss: Optional[float] = None,
        val_f1: Optional[float] = None,
        train_loss: Optional[float] = None,
        config: Optional[ModelConfig] = None,
        epoch: Optional[int] = None,
        cycle: Optional[int] = None,
        optimizer_state_dict: Optional[Dict] = None,
    ):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        with open(model_path, "wb") as model_path_fp:
            torch.save(self.state_dict(), model_path_fp)
        model_path_root = model_path.parent
        models_metadata_path = model_path_root / "metadata.json"
        models_metadata = {}
        if models_metadata_path.exists():
            with open(models_metadata_path, "r") as models_metadata_fp:
                models_metadata = json.load(models_metadata_fp)
        optimizer_path = None
        if optimizer_state_dict is not None:
            optimizer_path = (
                model_path_root / f"{model_path.stem}_optimizer{model_path.suffix}"
            )
            with open(optimizer_path, "wb") as optimizer_path_fp:
                torch.save(optimizer_state_dict, optimizer_path_fp)
        models_metadata.update(
            {
                model_path.stem: {
                    "optimizer_path": str(optimizer_path)
                    if optimizer_path
                    else optimizer_path,
                    "epoch": epoch,
                    "cycle": cycle,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "train_loss": train_loss,
                    "config": asdict(config) if config else config,
                }
            }
        )
        with open(models_metadata_path, "w") as models_metadata_fp:
            json.dump(models_metadata, models_metadata_fp)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # Apply Kaiming initialization for LeakyReLU
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.15
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr, batch, eps=None):
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        if eps == 0.0:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)  # Sample latent variable

        class_logits = self.classifier(
            global_mean_pool(z, batch)
        )  # Graph-level classification
        return z, mu, logvar, class_logits

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(
            logvar, min=-10, max=10
        )  # Clamp values to prevent extreme exponentiation
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index):
        # Predict adjacency matrix
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))

        # Predict edge attributes using an MLP
        row, col = edge_index
        edge_features = torch.cat(
            [z[row], z[col]], dim=-1
        )  # Concatenate node embeddings
        edge_attr_pred = self.edge_mlp(edge_features)
        edge_attr_pred = torch.sigmoid(
            edge_attr_pred
        )  # Assuming attributes are in [0,1]

        return adj_pred, edge_attr_pred

    def recon_loss(self, z, edge_index, edge_attr):
        adj_pred, edge_attr_pred = self.decode(z, edge_index)

        # Build adjacency ground truth
        adj_true = torch.zeros_like(adj_pred, dtype=torch.float32)
        adj_true[edge_index[0], edge_index[1]] = 1.0

        # Loss for adjacency matrix reconstruction (BCE Loss)
        adj_loss = F.binary_cross_entropy(adj_pred, adj_true)

        # Loss for edge attribute reconstruction (MSE Loss)
        edge_attr_pred_selected = edge_attr_pred
        edge_loss = F.mse_loss(edge_attr_pred_selected, edge_attr)
        # return adj_loss
        return 0.1 * adj_loss + edge_loss

    def kl_loss(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)  # Prevent extreme values
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class EnsembleEdgeVGAE:
    def __init__(self, model_paths: List[Path | str]) -> None:
        self.models: List[EdgeVGAE] = []
        self.val_losses: List[float] = []
        self.val_f1s: List[float] = []
        for model_path in model_paths:
            if isinstance(model_path, str):
                model_path = Path(model_path)
            model_path_root = model_path.parent
            model_metadata_path = model_path_root / "metadata.json"
            if not model_metadata_path.exists():
                continue
            model = EdgeVGAE.from_pretrained(model_path)
            self.models.append(model)
            with open(model_metadata_path, "r") as models_metadata_fp:
                models_metadata: Dict = json.load(models_metadata_fp)[model_path.stem]
                val_loss = models_metadata["val_loss"]
                val_f1 = models_metadata["val_f1"]
                self.val_losses.append(val_loss)
                self.val_f1s.append(val_f1)
