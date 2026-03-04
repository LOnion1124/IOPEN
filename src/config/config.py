import yaml
import argparse
import json
import torch

cfg = {}

with open('config.yaml', 'r') as file:
   cfg = yaml.safe_load(file)

if cfg['train']['dataset_path'] is not None:
   cam_path = cfg['train']['dataset_path'] + "camera.json"
   obj_path = cfg['train']['dataset_path'] + "models/models_info.json"

   with open(cam_path) as f:
      cam_cfg = json.load(f)
      cfg['cam'] = cam_cfg
   
   with open(obj_path) as f:
      obj_cfg = json.load(f)
      cfg['obj'] = obj_cfg['1']

parser = argparse.ArgumentParser()
args = parser.parse_known_args()


def get_device() -> torch.device:
   if torch.cuda.is_available():
      gpu_id = int(cfg['train'].get('gpu_id', 0))
      return torch.device(f"cuda:{gpu_id}")
   return torch.device("cpu")