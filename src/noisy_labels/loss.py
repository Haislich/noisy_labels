import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from noisy_labels.load_data import IndexedData


class NoisyCrossEntropyLoss(nn.Module):
    def __init__(self, p_noisy: float):
        super().__init__()
        self.p = p_noisy
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (
            1 - F.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1)
        )
        return (losses * weights).mean()


class OutlierDiscountingLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, labels):
        ce_loss = F.cross_entropy(pred, labels, reduction="none")
        pt = torch.exp(-ce_loss)

        # Focal loss component per down-weight outliers
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Outlier detection: high loss samples are likely outliers
        if len(ce_loss) > 1:
            loss_threshold = torch.quantile(ce_loss, 0.7)  # Top 30% losses
            outlier_mask = (ce_loss > loss_threshold).float()
        else:
            outlier_mask = torch.zeros_like(ce_loss)

        # Discount outliers
        discount_factor = 1.0 - 0.5 * outlier_mask

        return (focal_weight * ce_loss * discount_factor).mean()


class WeightedSymmetricCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        alpha: float = 0.1,
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, pred, labels, class_weights):
        ce = F.cross_entropy(pred, labels, reduction="none", weight=class_weights)

        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)

        label_one_hot = torch.zeros(pred.size()).to(pred.device)
        label_one_hot.scatter_(1, labels.view(-1, 1), 1)

        if class_weights is not None:
            weights_per_sample = class_weights[labels].view(-1, 1)  # shape [B, 1]
            rce = -torch.sum(
                weights_per_sample * label_one_hot * torch.log(pred_softmax), dim=1
            )
        else:
            rce = -torch.sum(label_one_hot * torch.log(pred_softmax), dim=1)

        loss = self.alpha * ce + self.beta * rce
        return loss.mean()


class SymmetricCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes: int = 6, alpha: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # CCE
        ce = F.cross_entropy(logits, targets, reduction="none")

        # RCE
        pred = F.softmax(logits, dim=1).clamp(min=1e-6, max=1 - 1e-6)
        one_hot = F.one_hot(targets, self.num_classes).float()
        rce = -(1 - one_hot) * torch.log(1 - pred)
        rce = rce.sum(dim=1)
        return (self.alpha * ce + self.beta * rce).mean()


class NCODLoss(nn.Module):
    past_embeddings: torch.Tensor
    centroids: torch.Tensor

    def __init__(
        self,
        dataset: Dataset[IndexedData],
        embedding_dimensions: int = 300,
        total_epochs: int = 150,
        lambda_consistency: float = 1.0,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embedding_dimensions = embedding_dimensions
        self.total_epochs = total_epochs
        self.lambda_consistency = lambda_consistency

        labels = [int(elem.y.item()) for elem in dataset]  # type: ignore
        self.num_elements = len(labels)
        self.num_classes = max(labels) + 1
        tmp_bins: list[list[int]] = [[] for _ in range(self.num_classes)]
        for idx, lab in enumerate(labels):
            tmp_bins[lab].append(idx)
        self.bins = [
            torch.as_tensor(b, dtype=torch.long, device=self.device) for b in tmp_bins
        ]

        self.u = nn.Parameter(torch.empty(self.num_elements, 1, device=self.device))
        nn.init.normal_(self.u, mean=1e-8, std=1e-9)

        self.register_buffer(
            "past_embeddings",
            torch.rand(
                self.num_elements, self.embedding_dimensions, device=self.device
            ),
        )

        self.register_buffer(
            "centroids",
            torch.rand(self.num_classes, self.embedding_dimensions, device=self.device),
        )

    def forward(
        self,
        *,
        logits: torch.Tensor,
        indexes: torch.Tensor,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        eps = 1e-6
        embeddings = F.normalize(embeddings, dim=1)
        self.past_embeddings[indexes] = embeddings.detach()

        if epoch == 0:
            with torch.no_grad():
                for c, idxs in enumerate(self.bins):
                    if idxs.numel():
                        self.centroids[c] = self.past_embeddings[idxs].mean(0)
        else:
            percent = int(max(1, min(100, 50 + 50 * (1 - epoch / self.total_epochs))))
            for c, idxs in enumerate(self.bins):
                if idxs.numel() == 0:
                    continue
                k = max(1, idxs.numel() * percent // 100)
                u_batch = self.u[idxs].squeeze(1)
                keep = torch.topk(u_batch, k, largest=False).indices
                selected = idxs[keep]
                self.centroids[c] = self.past_embeddings[selected].mean(0)

        centroids = F.normalize(self.centroids, dim=1)
        soft_labels = F.softmax(embeddings @ centroids.T, dim=1)
        probs = F.softmax(logits, dim=1)
        u_vals = torch.sigmoid(self.u[indexes]).squeeze(1)

        adjusted = (probs + u_vals[:, None] * soft_labels).clamp(min=eps)
        adjusted = adjusted / adjusted.sum(1, keepdim=True)

        hard_ce = (
            (1.0 - u_vals) * F.cross_entropy(logits, targets, reduction="none")
        ).mean()
        soft_ce = -(soft_labels * torch.log(adjusted)).sum(1).mean()
        consistency = F.mse_loss(adjusted, soft_labels)

        return hard_ce + soft_ce + self.lambda_consistency * consistency
