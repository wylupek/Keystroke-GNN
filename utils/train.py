import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split
import torch_geometric.loader as torchLoader

from utils import data_loader as dl
import pandas as pd


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


class SimpleGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(SimpleGraphDataset, self).__init__('.', None, None, None)
        self.data, self.slices = self.collate(data_list)  # Collate all data objects

    def __len__(self):
        return len(self.data.y)  # Number of graphs in the dataset




def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_train_split = 0.2  # 20% into test
    NUM_HIDDEN_DIMS = 64
    NUM_EPOCHS = 700
    file_names = ["../datasets/key_presses (1).tsv", "../datasets/key_presses (3).tsv", "../datasets/key_presses (4).tsv"]
    file_name = "../datasets/key_presses (2).tsv"
    positive_index = 2

    df_body = []

    # stuff we want to change
    mode = dl.load_char_mode.DROP
    rows_per_example = 30

    negative_datasets = [
        dl.load_data_object(filename, mode=mode, y=torch.tensor(0), rows_per_example=rows_per_example)
        for filename in file_names]
    negative_datasets.append(dl.load_data_object(file_name, mode=mode, y=torch.tensor(1), rows_per_example=rows_per_example))
    # positive_dataset = (dl.load_data_object(file_name, mode=mode, y=torch.tensor(1), rows_per_example=rows_per_example))





    training_dataset = []
    testing_dataset = []

    for negative_dataset in negative_datasets:
        train, test = train_test_split(negative_dataset, test_size=test_train_split)
        training_dataset.extend(train)
        testing_dataset.extend(test)
    # training_dataset_pos, testing_dataset_pos = train_test_split(positive_dataset, test_size=test_train_split)

    dataset = SimpleGraphDataset([e.to(device) for e in training_dataset])
    test_dataset = SimpleGraphDataset([e.to(device) for e in testing_dataset])

    model = LetterGNN(num_node_features=dataset.num_node_features, hidden_dim=NUM_HIDDEN_DIMS,
                      num_classes=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training over epochs
    data_loader = torchLoader.DataLoader(dataset, batch_size=32, shuffle=True)
    prev_loss = 100000
    for epoch in range(1, NUM_EPOCHS):
        model.train()
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)  # Forward pass
            loss = criterion(output, data.y)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model parameters
            total_loss += loss.item()
        loss = total_loss / len(data_loader)
        if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        if round(loss, 3) == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            break


    test_loader = torchLoader.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    correct = 0
    for data in test_loader:
        output = model(data.x, data.edge_index, data.batch)
        pred = output.argmax(dim=1)  # Get the index of the max log-probability
        # print(f"Correct {data.y[0]} Pred {pred[0]}")
        if pred != data.y:
            print(output[0])
        correct += (pred == data.y).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    train()