"""Module for loading data"""

import gzip
import json
import pickle
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset


class IndexedData(Data):
    idx: int

    def __init__(
        self,
        x: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        y: torch.Tensor | int | float | None = None,
        pos: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)


class IndexedSubset(Dataset):
    def __init__(self, subset: torch.utils.data.Subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        data = self.subset[i]
        data.idx = torch.tensor(i, dtype=torch.long)  # type: ignore
        return data


def dict_to_graph(g):
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    edge_attr = (
        torch.tensor(g["edge_attr"], dtype=torch.float) if g["edge_attr"] else None
    )
    y = torch.tensor(g["y"][0], dtype=torch.long) if g["y"] is not None else None
    return IndexedData(
        edge_index=edge_index, edge_attr=edge_attr, num_nodes=g["num_nodes"], y=y
    )


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


class GraphDataset(Dataset):
    def __init__(
        self,
        filename: Path | str,
        transform=add_zeros,
        pre_transform=None,
    ):
        self.filename = Path(filename)
        self.cache_path = self.filename.parent / f"{self.filename.stem}.pkl"
        # print(self.cache_path)
        # exit()

        super().__init__(None, transform, pre_transform)

        if self.cache_path.exists():
            with open(self.cache_path, "rb") as cache_file:
                self.graphs: list[IndexedData] = pickle.load(cache_file)
        else:
            self.graphs = self._build_graphs()
            with open(self.cache_path, "wb") as cache_file:
                pickle.dump(self.graphs, cache_file)

        self.num_graphs = len(self.graphs)

    def len(self):
        return self.num_graphs

    def get(self, idx):
        g = self.graphs[idx]
        return self.transform(g) if self.transform else g

    def _build_graphs(self):
        with gzip.open(self.filename, "rt", encoding="utf-8") as f:
            dicts = json.load(f)

        data_list = [dict_to_graph(d) for d in dicts]

        if self.pre_transform:
            data_list = [self.pre_transform(g) for g in data_list]

        return data_list
