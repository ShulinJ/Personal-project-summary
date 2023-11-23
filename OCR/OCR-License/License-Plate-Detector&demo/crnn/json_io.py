import json
from easydict import EasyDict as can_dict

def save_json_data(json_path, data_dict):
    with open(json_path, 'w+', encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)


def load_json_data(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data_dict = can_dict(json.load(f))
        return data_dict
