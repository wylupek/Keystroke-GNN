import torch
import pandas as pd

import letter_encoding
from torch_geometric.data import Data
from enum import Enum
from torch.nn.functional import one_hot as Torch_one_hot
import csv
from typing import Tuple, List

KEY_COL = "Key"


def process_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[int]]]:
    """
    Processes a DataFrame to compute average durations before and after key events,
    calculates mean values for grouped accelerometer data, and maps keys to vertex indices.

    :param df: pd.DataFrame:
        | KEY_COL - Key identifying each event.
        | Duration - Duration values for each event.
        | AccelX, AccelY, AccelZ - Accelerometer data.

    :return: Tuple:
            - pd.DataFrame:
                | KEY_COL: Encoded `KEY_COL` values.
                | AccelX, AccelY, AccelZ: Mean accelerometer data grouped by `KEY_COL`.
                | duration_before: Average duration of events before each key.
                | duration_after: Average duration of events after each key.
            - List[List[int], List[int]]:
                - The first list contains indices of starting keys (edge beginnings).
                - The second list contains indices of ending keys (edge endings).
    """
    avg_duration_before = {}
    avg_duration_after = {}
    key_pairs = ([], [])

    for i in range(1, len(df)):
        key_this = df.iloc[i][KEY_COL]
        dur_this = df.iloc[i]['Duration']
        key_before = df.iloc[i-1][KEY_COL]
        dur_before = df.iloc[i-1]['Duration']
        key_pairs[0].append(key_before)
        key_pairs[1].append(key_this)

        # Insert keys
        if not key_this in avg_duration_before:
            avg_duration_before[key_this] = (0, 0)
        if not key_this in avg_duration_after:
            avg_duration_after[key_this] = (0, 0)
        if not key_before in avg_duration_before:
            avg_duration_before[key_before] = (0, 0)
        if not key_before in avg_duration_after:
            avg_duration_after[key_before] = (0, 0)

        s, c = avg_duration_before[key_this]
        avg_duration_before[key_this] = (s + dur_this, c + 1)
        s, c = avg_duration_after[key_before]
        avg_duration_after[key_before] = (s + dur_before, c + 1)

    avg_duration_before = { k:(t[0]/t[1]) if t[1] != 0 else 0 for (k,t) in avg_duration_before.items() }
    avg_duration_after =  { k:(t[0]/t[1]) if t[1] != 0 else 0 for (k,t) in avg_duration_after.items()  }

    df = df[[KEY_COL, "AccelX","AccelY","AccelZ"]].groupby(KEY_COL).mean().reset_index()
    df["duration_before"] = df[KEY_COL].apply(lambda x: avg_duration_before.get(x))
    df["duration_after"] = df[KEY_COL].apply(lambda x: avg_duration_after.get(x))

    # THIS IS WHERE VERTEX NUMBERING HAPPENS
    # go over chars, get the row index of that char
    edge_beginnings = [df.index[df[KEY_COL] == c][0] for c in key_pairs[0]]
    edge_endings = [df.index[df[KEY_COL] == c][0] for c in key_pairs[1]]

    # now encode key as from letter_encoding (as int value, not python enum object)
    df[KEY_COL] = df[KEY_COL].apply(lambda x: letter_encoding.char_to_enum_value(x))

    # print(df, [edge_beginnings, edge_endings])
    return df, [edge_beginnings, edge_endings]


# enum to control behavior of below function
class load_char_mode(Enum):
    DROP = 0,
    INT = 1,
    ONE_HOT = 2


def create_data_obj(df, edges, y, mode=load_char_mode.INT):
    edge_index = torch.tensor(edges, dtype=torch.long)
    if mode == load_char_mode.DROP:
        node_attributes = torch.from_numpy(df.drop(columns=[KEY_COL, 'AccelX', 'AccelY', 'AccelZ']).values).float()
    
    elif mode == load_char_mode.INT:
        node_attributes = torch.from_numpy(df.values).float()

    elif mode == load_char_mode.ONE_HOT:
        keys = df[KEY_COL].to_list()
        key_tensor = torch.tensor(keys, dtype=torch.long)
        one_hot_keys = Torch_one_hot(key_tensor, num_classes=letter_encoding.AlphabetSize)
        features = torch.from_numpy(df.drop(columns=[KEY_COL]).values)
        node_attributes = torch.cat((features, one_hot_keys), dim=1).float()

    data = Data(x=node_attributes, edge_index=edge_index, y=y)

    return data


def load_data_object(filepath: str, y: torch.tensor, mode=load_char_mode.INT, rows_per_example=200) -> List[Data]:
    """
    Loads and processes data from a file to generate a list of PyTorch Geometric `Data` objects.
    :param filepath: str: Path to the input .tsv file
        | KeyPressTime
        | Duration
        | AccelX
        | AccelY
        | AccelZ
    :param y: torch.Tensor: Data owner label
    :param mode: mode for processing node attributes:
        - `load_char_mode.DROP`: Drops a specified key column (`KEY_COL`) during processing.
        - `load_char_mode.INT`: Uses all columns of the DataFrame as numerical features.
        - `load_char_mode.ONE_HOT`: Encodes the `KEY_COL` as a one-hot vector and concatenates
          it with other features.
    :param rows_per_example: int: Number of rows to include in one chunk of data
    :return: List[torch_geometric.data.Data]:
        A list of `Data` objects, where each object represents a processed chunk of data
        containing:
            - `x`: Node attributes as a tensor
            - `edge_index`: Edge indices tensor of shape [2, num_edges].
            - `y`: Data owner label
    """
    unprocessed = pd.read_csv(filepath, sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE)

    df_list = []
    edges_list = []
    
    n_splits = len(unprocessed)//rows_per_example
    i = 0
    for _ in range(n_splits - 1):
        d, e = process_df(unprocessed.iloc[i:i+rows_per_example])
        df_list.append(d)
        edges_list.append(e)
        i += rows_per_example

    # Last split
    d, e = process_df(unprocessed.iloc[i:])
    df_list.append(d)
    edges_list.append(e)

    data_objs = [create_data_obj(df, edges, y=y, mode=mode) for df, edges in zip(df_list, edges_list)]
    return data_objs


if __name__ == '__main__':
    # load_data_object('datasets/short.tsv', torch.tensor([1]), load_char_mode.DROP, 20)

    import sys
    if len(sys.argv) != 2:
        print("Pass csv as command line arg")
        exit(1)
    load_data_object(sys.argv[1], load_char_mode.ONE_HOT)
