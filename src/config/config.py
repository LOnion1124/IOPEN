import yaml
import argparse

cfg = {}

with open('config.yaml', 'r') as file:
   cfg = yaml.safe_load(file)

parser = argparse.ArgumentParser()
args = parser.parse_known_args()