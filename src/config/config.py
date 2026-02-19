import yaml
import argparse
import json

cfg = {}

with open('config.yaml', 'r') as file:
   cfg = yaml.safe_load(file)

if cfg['dataset_path'] is not None:
   cam_path = cfg['dataset_path'] + "camera.json"
   obj_path = cfg['dataset_path'] + "models/models_info.json"

   with open(cam_path) as f:
      cam_cfg = json.load(f)
      cfg['cam'] = cam_cfg
   
   with open(obj_path) as f:
      obj_cfg = json.load(f)
      cfg['obj'] = obj_cfg['1']

parser = argparse.ArgumentParser()
args = parser.parse_known_args()