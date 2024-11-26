from torch_geometric.loader import DataLoader
import torch
from utils.data_loader import LoadMode, load_from_str

from utils.train import SimpleGraphDataset, LetterGNN


def inference(user_id: str, content: str, model_path='', threshold=0.7,
              mode=LoadMode.ONE_HOT, rows_per_example=50, offset=1,
              hidden_dim=128) -> tuple[float, int]:
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
    :return: (accuracy, prediction)
    """
    device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    inference_dataset = load_from_str(content, y=torch.tensor([1]),
                                      mode=mode, rows_per_example=rows_per_example, offset=offset)
    inference_dataset = [e.to(device) for e in inference_dataset]
    inference_dataset = SimpleGraphDataset(inference_dataset)

    if model_path == '':
        model_path = f'models/{user_id}.pth'

    loaded_model = LetterGNN(num_node_features=inference_dataset.num_node_features, hidden_dim=hidden_dim,
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
