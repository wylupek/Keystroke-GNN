import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
from utils.data_loader import LoadMode
from collections import Counter

from utils.data_loader import create_data_obj, process_df
import pandas as pd
from torch_geometric.data import Data
import csv
from typing import List
from io import StringIO


def load_from_str(content: str, y: torch.tensor, mode=LoadMode.DROP, rows_per_example=200, offset=200) -> List[Data]:
    """
    Loads and processes data from a string to generate a list of PyTorch Geometric `Data` objects.
    :param content: str: string with .tsv content
            | key - Key identifying each event.
            | duration - Duration values for each event.
            | accel_x, accel_y, accel_z - Accelerometer data.
    :param y: data label
    :param mode: Mode for processing node attributes
    :param rows_per_example: Number of rows to include in one example
    :param offset: Number of rows between beginning of each example
    :return: List[torch_geometric.data.Data]:
        A list of `Data` objects, where each object represents a processed example
        containing:
            - `x`: Node attributes as a tensor
            - `edge_index`: Edge indices tensor of shape [2, num_edges].
            - `y`: Data owner label
    """
    unprocessed = pd.read_csv(StringIO(content), sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE)

    df_list = []
    edges_list = []

    i = 0
    while i+rows_per_example < len(unprocessed):
        d, e = process_df(unprocessed.iloc[i:i+rows_per_example])
        df_list.append(d)
        edges_list.append(e)
        i += offset

    # Last split
    d, e = process_df(unprocessed.iloc[i:])
    df_list.append(d)
    edges_list.append(e)

    data_objs = [create_data_obj(df, edges, y=y, mode=mode) for df, edges in zip(df_list, edges_list)]
    return data_objs


class SimpleGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(SimpleGraphDataset, self).__init__('.', None, None, None)
        self.data, self.slices = self.collate(data_list)  # Collate all data objects

    def __len__(self):
        return len(self.data.y)  # Number of graphs in the dataset

    def statistics(self) -> str:
        class_counts = Counter(self.data.y.cpu().numpy())
        return " | ".join([f"{item_class}: {count}" for item_class, count in
                           class_counts.items()]) + f" | Total: {sum(class_counts.values())}"


class LetterGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_layers=2):
        super(LetterGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(num_node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = x.float()
            edge_index = edge_index.long()
            x = conv.forward(x, edge_index)
            x = F.relu(x)

        # idk maybe use a different pool method
        x = pyg_nn.global_mean_pool(x, batch)

        # classify
        x = self.fc(x)

        return x


def inference(user_id: str, content: str, model_path='', threshold=0.7,
              mode=LoadMode.DROP, rows_per_example=30, offset=1,
              hidden_dim=64, num_node_features=2) -> tuple[float, int]:
    """
        Train and save the model
        :param user_id: user_id for positive labels
        :param content: .tsv content
        :param model_path: Path to save the model. Leave default to save at ./models/<user_id>.pth
        :param threshold: Threshold for positive prediction
        :param mode: Mode for processing node attributes
        :param rows_per_example: Number of key presses per example
        :param offset: Number of rows between beginning of each example
        :param hidden_dim: hidden dimension
        :param num_node_features: Number of node features (avg_before and avg_after - 2)
        :return: (accuracy, prediction)
        """
    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    inference_dataset = load_from_str(content, y=torch.tensor([1]),
                                      mode=mode, rows_per_example=rows_per_example, offset=offset)
    inference_dataset = [e.to(device) for e in inference_dataset]
    inference_dataset = SimpleGraphDataset(inference_dataset)

    if model_path == '':
        model_path = f'models/{user_id}.pth'

    loaded_model = LetterGNN(num_node_features=num_node_features, hidden_dim=hidden_dim,
                             num_classes=2).to(device)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=True))
    loaded_model.eval()


    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
    total_positives = 0
    with torch.no_grad():
        for data in inference_loader:
            data = data.to(device)
            output = loaded_model(data.x, data.edge_index, data.batch)
            total_positives += output.argmax(dim=1)
    accuracy = float(total_positives / len(inference_dataset))

    if accuracy < threshold:
        return accuracy, 0
    return accuracy, 1


if __name__ == '__main__':
    with open("../datasets/inference/user2.tsv", "r", encoding="utf-8") as file:
        tsv_content = file.read()

    print(inference("user3", tsv_content, model_path='../models/experimental.pth', threshold=0.5))
