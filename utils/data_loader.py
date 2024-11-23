import torch
from torch.nn.functional import one_hot as oh
from torch_geometric.data import Data
from utils import letter_encoding

import sqlite3
import pandas as pd
import csv
import random
from enum import Enum
from typing import Tuple, List



class LoadMode(Enum):
    """
    Enum to control behavior of dataloader
    `load_char_mode.DROP`: Drops a key column during processing.
    `load_char_mode.INT`: Uses all columns of the DataFrame as numerical features.
    `load_char_mode.ONE_HOT`: Encodes the key column as a one-hot vector and concatenates
        it with other features.
    """
    DROP = 0,
    INT = 1,
    ONE_HOT = 2


def create_data_obj(df, edges, y, mode=LoadMode.INT) -> Data:
    edge_index = torch.tensor(edges, dtype=torch.long)
    if mode == LoadMode.DROP:
        node_attributes = torch.from_numpy(df.drop(columns=["key", 'accel_x', 'accel_y', 'accel_z']).values).float()

    elif mode == LoadMode.INT:
        node_attributes = torch.from_numpy(df.values).float()

    else: # mode == LoadMode.ONE_HOT:
        keys = df["key"].to_list()
        key_tensor = torch.tensor(keys, dtype=torch.long)
        one_hot_keys = oh(key_tensor, num_classes=letter_encoding.AlphabetSize)
        features = torch.from_numpy(df.drop(columns=["key"]).values)
        node_attributes = torch.cat((features, one_hot_keys), dim=1).float()

    data = Data(x=node_attributes, edge_index=edge_index, y=y)

    return data


def process_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[int]]]:
    """
    Processes a DataFrame to compute average durations before and after key events,
    calculates mean values for grouped accelerometer data, and maps keys to vertex indices.
    :param df: pd.DataFrame:
        | key - Key identifying each event.
        | duration - Duration values for each event.
        | accel_x, accel_y, accel_z - Accelerometer data.
    :return: Tuple:
            - pd.DataFrame:
                | key: Encoded key values.
                | accel_x, accel_y, accel_z: Mean accelerometer data grouped by key.
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
        key_this = df.iloc[i]["key"]
        dur_this = df.iloc[i]['duration']
        key_before = df.iloc[i-1]["key"]
        dur_before = df.iloc[i-1]['duration']
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

    df = df[["key", "accel_x","accel_y","accel_z"]].groupby("key").mean().reset_index()
    df["duration_before"] = df["key"].apply(lambda x: avg_duration_before.get(x))
    df["duration_after"] = df["key"].apply(lambda x: avg_duration_after.get(x))

    # THIS IS WHERE VERTEX NUMBERING HAPPENS
    # go over chars, get the row index of that char
    edge_beginnings = [df.index[df["key"] == c][0] for c in key_pairs[0]]
    edge_endings = [df.index[df["key"] == c][0] for c in key_pairs[1]]

    # now encode key as from letter_encoding (as int value, not python enum object)
    df["key"] = df["key"].apply(lambda x: letter_encoding.char_to_enum_value(x))

    # print(df, [edge_beginnings, edge_endings])
    return df, [edge_beginnings, edge_endings]


def load_from_file(filepath: str, y: torch.tensor, mode=LoadMode.INT, rows_per_example=200) -> List[Data]:
    """
    Loads and processes data from a file to generate a list of PyTorch Geometric `Data` objects.
    :param filepath: str: Path to the input .tsv file
            | key - Key identifying each event.
            | duration - Duration values for each event.
            | accel_x, accel_y, accel_z - Accelerometer data.
    :param y: data label
    :param mode: Mode for processing node attributes
    :param rows_per_example: Number of rows to include in one example
    :return: List[torch_geometric.data.Data]:
        A list of `Data` objects, where each object represents a processed example
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


def get_positive_examples(conn: sqlite3.Connection, user_id: str,
                          rows_per_example=200, mode=LoadMode.DROP) -> List[Data]:
    """
    Loads and processes data from a database to generate a list of PyTorch Geometric `Data` objects.
    :param conn: Database connection
    :param user_id: user_id of positive examples
    :param mode: Mode for processing node attributes
    :param rows_per_example: Number of rows to include in one example
    :return: List[torch_geometric.data.Data]:
        A list of `Data` objects, where each object represents a processed examples containing:
            - `x`: Node attributes as a tensor
            - `edge_index`: Edge indices tensor of shape [2, num_edges].
            - `y`: tensor(1)
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM key_press
        WHERE user_id = ?
        ORDER BY press_time
    """, (user_id,))
    rows = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    df_list = []
    edges_list = []
    n_splits = len(rows) // rows_per_example
    i = 0
    for _ in range(n_splits - 1):
        d, e = process_df(rows[i:i + rows_per_example])
        df_list.append(d)
        edges_list.append(e)
        i += rows_per_example

    # Last split
    d, e = process_df(rows[i:])
    df_list.append(d)
    edges_list.append(e)

    data_objs = [create_data_obj(df, edges, y=torch.tensor(1), mode=mode) for df, edges in zip(df_list, edges_list)]
    return data_objs


def get_negative_examples(conn: sqlite3.Connection, user_id: str, num_examples: int,
                          rows_per_example=200, mode=LoadMode.DROP) -> List[Data]:
    """
    Loads and processes data from a database to generate a list of PyTorch Geometric `Data` objects.
    :param conn: Database connection
    :param user_id: user_id of positive examples
    :param mode: Mode for processing node attributes
    :param rows_per_example: Number of rows to include in one example
    :param num_examples: Number of negative examples to include
    :return: List[torch_geometric.data.Data]:
        A list of `Data` objects, where each object represents a processed examples containing:
            - `x`: Node attributes as a tensor
            - `edge_index`: Edge indices tensor of shape [2, num_edges].
            - `y`: tensor(0)
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM key_press
        WHERE user_id != ?
        ORDER BY press_time
    """, (user_id,))
    rows = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    total_rows = len(rows)
    starting_indices = set()
    index = 0
    while index <= total_rows - rows_per_example:
        if rows.iloc[index]['user_id'] == rows.iloc[index + rows_per_example - 1]['user_id'] and \
                rows.iloc[index]['date'] == rows.iloc[index + rows_per_example - 1]['date']:
                starting_indices.add(index)
                index += rows_per_example
        else:
            index += 1

    df_list = []
    edges_list = []
    total_starting_indices = len(starting_indices)
    if num_examples >= total_starting_indices:
        if num_examples > total_starting_indices:
            print(f"Warning: expected {num_examples} examples, but got {total_starting_indices}")
        for starting_index in starting_indices:
            d, e = process_df(rows[starting_index:starting_index + rows_per_example])
            df_list.append(d)
            edges_list.append(e)

    else:
        selected_indices = random.sample(sorted(starting_indices), num_examples)
        for starting_index in selected_indices:
            d, e = process_df(rows[starting_index:starting_index + rows_per_example])
            df_list.append(d)
            edges_list.append(e)

    data_objs = [create_data_obj(df, edges, y=torch.tensor(0), mode=mode) for df, edges in zip(df_list, edges_list)]
    return data_objs


def load_from_db(user_id: str, positive_negative_ratio: float,
                 mode=LoadMode.INT, rows_per_example=200) -> List[Data]:
    """
    Loads and processes data from a database to generate a list of PyTorch Geometric `Data` objects.
    :param user_id: user_id of positive examples
    :param mode: Mode for processing node attributes
    :param rows_per_example: Number of rows to include in one example
    :param positive_negative_ratio: Positive to negative examples ratio
    :return: List[torch_geometric.data.Data]:
        A list of `Data` objects, where each object represents a processed examples containing:
            - `x`: Node attributes as a tensor
            - `edge_index`: Edge indices tensor of shape [2, num_edges].
            - `y`: Data label
    """

    try:
        conn = sqlite3.connect('keystroke_data.sqlite')
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return []


    positive_examples = get_positive_examples(conn, user_id, rows_per_example, mode)
    negative_examples = (get_negative_examples(conn, user_id,
                                               int(len(positive_examples) / positive_negative_ratio),
                                               rows_per_example, mode))
    return positive_examples + negative_examples


if __name__ == '__main__':
    # load_from_file('datasets/user1.tsv', torch.tensor([1]), LoadMode.DROP, 20)
    load_from_db("user2", 0.5, mode=LoadMode.DROP, rows_per_example=10)
