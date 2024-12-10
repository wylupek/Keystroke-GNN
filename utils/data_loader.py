import torch
from torch.nn.functional import one_hot as oh
from torch_geometric.data import Data

from utils import letter_encoding

import sqlite3
import pandas as pd
import numpy as np
import csv
from enum import Enum
from typing import Tuple, List
from io import StringIO


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


def create_data_obj(df: pd.DataFrame, edges: list, y: torch.tensor, mode=LoadMode.ONE_HOT, use_accel=False) -> Data:
    """
    Creates Data object from torch_geometric.data
    :param df: Dataframe with
        | key: Encoded key values.
        | accel_x, accel_y, accel_z: Mean accelerometer data grouped by key.
        | duration_before: Average duration of events before each key.
        | duration_after: Average duration of events after each key.
    :param edges: List[List[int], List[int]]:
        - The first list contains indices of starting keys (edge beginnings).
        - The second list contains indices of ending keys (edge endings).
    :param y: data label
    :param mode: Mode for processing node attributes
    :param use_accel: whether to use accelerometer data
    :return: `Data` object that represents a processed example containing:
        - `x`: Node attributes as a tensor.
        - `edge_index`: Edge indices tensor of shape [2, num_edges].
        - `y`: tensor(y).
    """

    edge_index = torch.tensor(edges, dtype=torch.long)
    if not use_accel:
        df = df.drop(columns=['accel_x', 'accel_y', 'accel_z'])

    if mode == LoadMode.ONE_HOT:
        keys = df["key"].to_list()
        key_tensor = torch.tensor(keys, dtype=torch.long)
        one_hot_keys = oh(key_tensor, num_classes=letter_encoding.AlphabetSize)
        features = torch.from_numpy(df.drop(columns=["key"]).values)
        node_attributes = torch.cat((features, one_hot_keys), dim=1).float()
    elif mode == LoadMode.DROP:
        node_attributes = torch.from_numpy(df.drop(columns=["key"]).values).float()
    else: # mode == LoadMode.INT:
        node_attributes = torch.from_numpy(df.values).float()

    return Data(x=node_attributes, edge_index=edge_index, y=y)


def create_data_obj_extra_feature(df: pd.DataFrame, edges: list, y: torch.tensor, mat, mode=LoadMode.ONE_HOT, use_accel=False, ) -> Data:
    """
    Creates Data object from torch_geometric.data
    :param df: Dataframe with
        | key: Encoded key values.
        | accel_x, accel_y, accel_z: Mean accelerometer data grouped by key.
        | duration_before: Average duration of events before each key.
        | duration_after: Average duration of events after each key.
    :param edges: List[List[int], List[int]]:
        - The first list contains indices of starting keys (edge beginnings).
        - The second list contains indices of ending keys (edge endings).
    :param y: data label
    :param mode: Mode for processing node attributes
    :param use_accel: whether to use accelerometer data
    :return: `Data` object that represents a processed example containing:
        - `x`: Node attributes as a tensor.
        - `edge_index`: Edge indices tensor of shape [2, num_edges].
        - `y`: tensor(y).
    """

    edge_index = torch.tensor(edges, dtype=torch.long)
    if not use_accel:
        df = df.drop(columns=['accel_x', 'accel_y', 'accel_z'])

    if mode == LoadMode.ONE_HOT:
        keys = df["key"].to_list()
        key_tensor = torch.tensor(keys, dtype=torch.long)
        one_hot_keys = oh(key_tensor, num_classes=letter_encoding.AlphabetSize)
        
        input_rows = []
        for i, key in enumerate(keys):
            # Extract the slice from mat corresponding to the key
            mat_slice = mat[key]  # Shape: (AlphabetSize, 2)
            mat_flattened = mat_slice.flatten()  # Shape: (AlphabetSize * 2)
            
            # Concatenate with the one-hot encoded row
            one_hot_row = one_hot_keys[i].to(torch.float32)  # Ensure the same dtype
            input_row = torch.cat((one_hot_row, mat_flattened))  # Concatenate
            input_rows.append(input_row)

        # Stack all rows to form the final input tensor
        node_attributes = torch.stack(input_rows).float()  # Shape: (len(keys), num_classes + AlphabetSize * 2)

        features = torch.from_numpy(df.drop(columns=["key"]).values)
        node_attributes = torch.cat((features, one_hot_keys), dim=1).float()
    
    elif mode == LoadMode.DROP:
        node_attributes = torch.from_numpy(df.drop(columns=["key"]).values).float()
    else: # mode == LoadMode.INT:
        raise RuntimeError("not implemented")



    return Data(x=node_attributes, edge_index=edge_index, y=y)


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



def process_df_with_per_letter_average(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[List[int]]]:
    """
    Processes a DataFrame to compute average durations before and after key events 
    for each letter separately, filling with zero otherwise,
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
    # the key to avg_duration is just tuples: (before_key, this_key) (this_key, after_key)
    # values is (duration sum, count)
    avg_duration = {}
    key_pairs = ([], [])

    for i in range(1, len(df)):
        key_this = df.iloc[i]["key"]
        dur_this = df.iloc[i]['duration']
        key_before = df.iloc[i-1]["key"]
        key_pairs[0].append(key_before)
        key_pairs[1].append(key_this)

        letter_pair = (key_before, key_this) 
        # Insert keys
        if not letter_pair in avg_duration:
            avg_duration[letter_pair] = (0, 0)
            

        s, c = avg_duration[letter_pair]
        avg_duration[letter_pair] = (s + dur_this, c + 1)

    # the key is the second letter in the pair
    durs_before_this_key = { 
        letter_after : ( dur_sum / dur_count ) 
        if dur_count != 0 else 0 
        for ((letter_before, letter_after), (dur_sum, dur_count)) 
        in avg_duration.items() 
    }
    durs_after_this_key = { 
        letter_before : ( dur_sum / dur_count ) 
        if dur_count != 0 else 0 
        for ((letter_before, letter_after), (dur_sum, dur_count)) 
        in avg_duration.items() 
    }

    df = df[["key", "accel_x","accel_y","accel_z"]].groupby("key").mean().reset_index()

    # deepest each column is time before, time after
    times_matrix = torch.zeros([letter_encoding.AlphabetSize, letter_encoding.AlphabetSize, 2])

    for ((letter_before, letter_after), (dur_sum, dur_count)) in avg_duration.items():
        before_encoded = letter_encoding.char_to_enum_value(letter_before)
        after_encoded = letter_encoding.char_to_enum_value(letter_after)
        

        times_matrix[after_encoded][before_encoded][0] = ( dur_sum / dur_count )
        times_matrix[before_encoded][after_encoded][1] = ( dur_sum / dur_count )

    
    # THIS IS WHERE VERTEX NUMBERING HAPPENS
    # go over chars, get the row index of that char
    edge_beginnings = [df.index[df["key"] == c][0] for c in key_pairs[0]]
    edge_endings = [df.index[df["key"] == c][0] for c in key_pairs[1]]

    # now encode key as from letter_encoding (as int value, not python enum object)
    df["key"] = df["key"].apply(lambda x: letter_encoding.char_to_enum_value(x))

    # print(df, [edge_beginnings, edge_endings])
    return df, [edge_beginnings, edge_endings], (times_matrix)


def load_from_file(filepath: str, y: torch.tensor, mode=LoadMode.ONE_HOT, rows_per_example=50, offset=10) -> List[Data]:
    """
    Loads and processes data from a file to generate a list of PyTorch Geometric `Data` objects.
    :param filepath: str: Path to the input .tsv file
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
    unprocessed = pd.read_csv(filepath, sep="\t", encoding='utf-8', quoting=csv.QUOTE_NONE)

    df_list = []
    edges_list = []

    i = 0
    while i + rows_per_example <= len(unprocessed):
        d, e = process_df(unprocessed.iloc[i : i + rows_per_example])
        df_list.append(d)
        edges_list.append(e)
        i += offset

    return [create_data_obj(df, edges, y=y, mode=mode) for df, edges in zip(df_list, edges_list)]


def load_from_str(content: str, y: torch.tensor, mode=LoadMode.ONE_HOT, rows_per_example=50, offset=10) -> List[Data]:
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
    while i + rows_per_example <= len(unprocessed):
        d, e = process_df(unprocessed.iloc[i : i + rows_per_example])
        df_list.append(d)
        edges_list.append(e)
        i += offset

    return [create_data_obj(df, edges, y=y, mode=mode) for df, edges in zip(df_list, edges_list)]


def get_user_examples(conn: sqlite3.Connection, user_id: int, y: torch.tensor, 
                      mode=LoadMode.ONE_HOT, rows_per_example=50, offset=10, agg_time=True) -> List[Data]:
    """
    Loads and processes data from a database to generate a list of PyTorch Geometric `Data` objects.
    :param conn: Database connection
    :param user_id: user_id of user for which we want to get the examples
    :param y: data label
    :param mode: Mode for processing node attributes
    :param rows_per_example: Number of rows to include in one example
    :param offset: Number of rows between beginning of each example
    :param agg_time: For input node features use avg time before and after. If set to false, 
        a tensor of per letter before and after averages will be used.  
    :return: List[torch_geometric.data.Data]:
        A list of `Data` objects, where each object represents a processed examples containing:
            - `x`: Node attributes as a tensor.
            - `edge_index`: Edge indices tensor of shape [2, num_edges].
            - `y`: tensor(y).
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

    i = 0
    while i + rows_per_example <= len(rows):
        if rows.iloc[i]['timestamp'] == rows.iloc[i + rows_per_example - 1]['timestamp']:
            # print(f"{user_id} [{i} : {i + rows_per_example}]")
            if agg_time:
                d, e = process_df(rows.iloc[i : i + rows_per_example])
            else:
                d, e, extra_feature_matrix = process_df_with_per_letter_average(rows.iloc[i : i + rows_per_example])

            df_list.append(d)
            edges_list.append(e)
            i += offset
        else:
            i += 1

    if agg_time:
        return [create_data_obj(df, edges, y=torch.tensor(y), mode=mode) for df, edges in zip(df_list, edges_list)]

    else:
        return [create_data_obj_extra_feature(df, edges, y=torch.tensor(y), mode=mode, mat=extra_feature_matrix) for df, edges in zip(df_list, edges_list)]


def load_from_db(database_path: str, user_id: int, positive_negative_ratio: float,
                 mode=LoadMode.ONE_HOT, rows_per_example=50, offset=10) -> Tuple[List[Data], List[List[Data]]]:
    """
    Loads and processes data from a database to generate a list of PyTorch Geometric `Data` objects.
    :param database_path: Path to database
    :param user_id: user_id of positive examples
    :param mode: Mode for processing node attributes
    :param rows_per_example: Number of rows to include in one example. Set to 0 to load all examples
    :param positive_negative_ratio: Positive to negative examples ratio
    :param offset: Number of rows between beginning of each example
    :return: List[torch_geometric.data.Data]:
        A list of `Data` objects, where each object represents a processed examples containing:
            - `x`: Node attributes as a tensor
            - `edge_index`: Edge indices tensor of shape [2, num_edges].
            - `y`: Data label
    """

    try:
        conn = sqlite3.connect(database_path)
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return [], []

    positive_examples = get_user_examples(conn, user_id, y=1,
                                          mode=mode, rows_per_example=rows_per_example, offset=offset)

    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT user_id
        FROM key_press
        where user_id != ?
    """, (user_id,))
    other_users = [i[0] for i in cursor.fetchall()]

    negative_examples = []
    if positive_negative_ratio == 0:
        for user in other_users:
            negative_examples.extend(
                get_user_examples(conn, user, y=0,
                                  mode=mode, rows_per_example=rows_per_example, offset=offset)
            )

    else:
        neg_per_user = int((len(positive_examples) / positive_negative_ratio) // len(other_users)) + rows_per_example*2
        for user in other_users:
            neg_list = get_user_examples(conn, user, y=0,
                                         mode=mode, rows_per_example=rows_per_example, offset=offset)

            if neg_per_user >= len(neg_list):
                negative_examples.append(neg_list)
                continue

            step = (len(neg_list) - 1) / (neg_per_user - 1)
            selected_indices = [round(i * step) for i in range(neg_per_user)]
            negative_examples.append([neg_list[i] for i in selected_indices])

    print(f"*** DATA LOADER INFO ***\n"
          f"Positives: {len(positive_examples)}\n"
          f"Negatives: {sum(len(x) for x in negative_examples)} {[len(x) for x in negative_examples]}"
          f"************************\n")

    return positive_examples, negative_examples


if __name__ == '__main__':
    load_from_db('./keystroke_data.sqlite', 'user4', positive_negative_ratio=1,
                 rows_per_example=100, offset=30)
