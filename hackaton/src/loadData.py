# import gzip
# import json
# import torch
# from torch_geometric.data import Dataset, Data
# import os
# from tqdm import tqdm
# from torch_geometric.loader import DataLoader

# class GraphDataset(Dataset):
#     def __init__(self, filename, transform=None, pre_transform=None):
#         self.raw = filename
#         self.graphs = self.loadGraphs(self.raw)
#         super().__init__(None, transform, pre_transform)

#     def len(self):
#         return len(self.graphs)

#     def get(self, idx):
#         return self.graphs[idx]

#     @staticmethod
#     def loadGraphs(path):
#         print(f"Loading graphs from {path}...")
#         print("This may take a few minutes, please wait...")
#         with gzip.open(path, "rt", encoding="utf-8") as f:
#             graphs_dicts = json.load(f)
#         graphs = []
#         for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
#             graphs.append(dictToGraphObject(graph_dict))
#         return graphs


# def dictToGraphObject(graph_dict):
#     edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
#     edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
#     num_nodes = graph_dict["num_nodes"]
#     y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
#     return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)
import gzip
import json
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset


def dict_to_graph(g):
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    edge_attr = (
        torch.tensor(g["edge_attr"], dtype=torch.float) if g["edge_attr"] else None
    )
    y = torch.tensor(g["y"][0], dtype=torch.long) if g["y"] is not None else None
    return Data(
        edge_index=edge_index, edge_attr=edge_attr, num_nodes=g["num_nodes"], y=y
    )


class GraphDataset(Dataset):
    """
    Exactly the same public API as before.
    A single .pt file is written next to <filename> the first time you touch the dataset.
    """

    CACHE_EXT = ".pt"  # e.g. 'train.json.gz.pt'

    def __init__(self, filename, transform=None, pre_transform=None, use_cache=True):
        self.raw_path = Path(filename)
        self.cache_path = self.raw_path.with_suffix(
            self.raw_path.suffix + self.CACHE_EXT
        )

        # Let PyG create self.transform / self.pre_transform first
        super().__init__(None, transform, pre_transform)

        # --- 1.  Load or build -------------------------------------------------
        if use_cache and self.cache_path.exists():
            self.graphs = torch.load(self.cache_path, weights_only=False)
        else:
            self.graphs = self._build_graphs()
            if use_cache:
                torch.save(self.graphs, self.cache_path)

        self.num_graphs = len(self.graphs)

    # ---------- torch_geometric.Dataset interface ----------------------------
    def len(self):
        return self.num_graphs

    def get(self, idx):
        g = self.graphs[idx]
        return self.transform(g) if self.transform else g

    # ---------- helpers ------------------------------------------------------
    def _build_graphs(self):
        """Parse the gzip → list[Data].  Runs only the first time."""
        with gzip.open(self.raw_path, "rt", encoding="utf-8") as f:
            dicts = json.load(f)

        data_list = [dict_to_graph(d) for d in dicts]

        if self.pre_transform:
            data_list = [self.pre_transform(g) for g in data_list]

        return data_list

        return data_list
