import os
import torch
import json
import numpy as np

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def moving_average(arr, window_size):
    cumsum = np.cumsum(arr, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

def save_dict_to_json(data, filename):
    # 确保文件名以.json结尾
    if not filename.endswith('.json'):
        filename += '.json'

    # 打开文件以写入
    with open(filename, 'w', encoding='utf-8') as f:
        # 将字典转换为JSON格式并写入文件
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_model_path(model_name):
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'results', model_name)
    model_path = os.path.join(model_dir, 'model')
    return model_path

def save_model_and_train_result(model, result, model_name):
    cwd = os.getcwd()
    print(cwd)
    model_dir = os.path.join(cwd, 'results', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'model')
    torch.save(model.state_dict(), model_path)
    result_path = os.path.join(model_dir, 'result.json')
    save_dict_to_json(result, result_path)
