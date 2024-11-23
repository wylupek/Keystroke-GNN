import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split
import torch_geometric.loader as torchLoader

# import utils.data_loader as dl
import pandas as pd
from collections import Counter
from utils.data_loader import load_from_db, LoadMode


# Define a simple GCN model
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

    def statistics(self) -> str:
        class_counts = Counter(self.data.y.cpu().numpy())
        return " | ".join([f"{item_class}: {count}" for item_class, count in
                           class_counts.items()]) + f" | Total: {sum(class_counts.values())}"


def train(database_path: str, user_id: str, mode: LoadMode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_train_split = 0.2  # 20% into test
    NUM_HIDDEN_DIMS = 64
    NUM_EPOCHS = 1000
    rows_per_example = 150

    examples = load_from_db(database_path=database_path, user_id=user_id, positive_negative_ratio=0,
                               mode=mode, rows_per_example=rows_per_example)

    train_pos, test_pos = train_test_split([ex for ex in examples if ex['y'] == 1],
                                   test_size=test_train_split)
    train_neg, test_neg = train_test_split([ex for ex in examples if ex['y'] == 0],
                                           test_size=test_train_split)
    train_examples = train_pos + train_neg
    test_examples = test_pos + test_neg

    del examples, train_pos, train_neg, test_pos, test_neg

    train_examples = SimpleGraphDataset([e.to(device) for e in train_examples])
    test_examples = SimpleGraphDataset([e.to(device) for e in test_examples])

    # print("Train dataset statistics: ", train_examples.statistics())
    # print("Test dataset statistics:  ", test_examples.statistics())

    model = LetterGNN(num_node_features=train_examples.num_node_features, hidden_dim=NUM_HIDDEN_DIMS,
                      num_classes=train_examples.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # # Training over epochs
    data_loader = torchLoader.DataLoader(train_examples, batch_size=32, shuffle=True)
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


    test_loader = torchLoader.DataLoader(test_examples, batch_size=1, shuffle=False)

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
    train("../keystroke_data.sqlite", "user1", LoadMode.DROP)
    # train()