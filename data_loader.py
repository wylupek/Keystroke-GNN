import torch
import pandas as pd
import letter_encoding
from torch_geometric.data import Data
from enum import Enum
from torch.nn.functional import one_hot as Torch_one_hot
from sklearn.model_selection import train_test_split

KEY_COL = "Key"
# load data into dataframe
# with columns: key avg_duration_before avg_duration_after accel_x accel_y accel_z
def process_df(df):
    # key : (sum , count)
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
            
        # insert keys
        if not key_this in avg_duration_before:
            avg_duration_before[key_this] = (0,0)
        if not key_this in avg_duration_after:
            avg_duration_after[key_this] = (0,0)
        if not key_before in avg_duration_before:
            avg_duration_before[key_before] = (0,0)
        if not key_before in avg_duration_after:
            avg_duration_after[key_before] = (0,0)

        s, c = avg_duration_before[key_this]
        avg_duration_before[key_this] = (s+dur_this , c+1)

        s, c = avg_duration_after[key_before]
        avg_duration_after[key_before] = (s+dur_before , c+1)

    avg_duration_before = { k:(t[0]/t[1]) if t[1] != 0 else 0 for (k,t) in avg_duration_before.items() }
    avg_duration_after =  { k:(t[0]/t[1]) if t[1] != 0 else 0 for (k,t) in avg_duration_after.items()  }

    df = df[[KEY_COL, "accel_x","accel_y","accel_z"]].groupby(KEY_COL).mean().reset_index()
    df["duration_before"] = df[KEY_COL].apply(lambda x: avg_duration_before.get(x))
    df["duration_after"] = df[KEY_COL].apply(lambda x: avg_duration_after.get(x))
    
    # THIS IS WHERE VERTEX NUMBERING HAPPENS
    # go over chars, get the row index of that char
    edge_beginnings = [df.index[df[KEY_COL] == c][0] for c in key_pairs[0]]
    edge_endings = [df.index[df[KEY_COL] == c][0] for c in key_pairs[1]]


    # now encode key as from letter_encoding (as int value, not python enum object)
    df[KEY_COL] = df[KEY_COL].apply(lambda x: letter_encoding.char_to_enum_value(x))

    return df, [edge_beginnings, edge_endings]


# enum to control behavior of below function
class load_char_mode(Enum):
    DROP = 0,
    INT = 1,
    ONE_HOT = 2

def create_data_obj(df, edges, mode = load_char_mode.INT):
    edge_index = torch.tensor(edges, dtype=torch.long)
    if mode == load_char_mode.DROP:
        node_attributes = torch.from_numpy(df.drop(columns=[KEY_COL]).values).float()
    
    elif mode == load_char_mode.INT:
        node_attributes = torch.from_numpy(df.values).float()

    elif mode == load_char_mode.ONE_HOT:
        keys = df[KEY_COL].to_list()
        key_tensor = torch.tensor(keys, dtype=torch.long)
        one_hot_keys = Torch_one_hot(key_tensor, num_classes=letter_encoding.AlphabetSize)
        features = torch.from_numpy(df.drop(columns=[KEY_COL]).values)
        node_attributes = torch.cat((features, one_hot_keys), dim=1).float()

    data = Data(x=node_attributes, edge_index=edge_index)

    return data

# returns a torch_geometric.data.Data object
def load_data_object(filepath, mode = load_char_mode.INT, test_split=0.2):
    unprocessed = pd.read_csv(filepath, sep="\t")
    train_df, test_df = train_test_split(unprocessed, test_size=test_split, random_state=42)


    train_df, train_edges = process_df(train_df)
    test_df, test_edges = process_df(test_df)

    train_data = create_data_obj(train_df, train_edges, mode)
    test_data = create_data_obj(test_df, test_edges, mode)

    return train_data, test_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Pass csv as command line arg")
        exit(1)
    load_data_object(sys.argv[1], load_char_mode.ONE_HOT)
