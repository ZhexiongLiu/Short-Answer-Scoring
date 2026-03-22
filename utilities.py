import pandas as pd
import json

def get_json_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data)
    return df
