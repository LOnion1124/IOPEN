import torch

input_path = "D:/projects/IOPEN/result/train/checkpoint_epoch_0000.pth"
output_path = "D:/projects/IOPEN/models/IOPEN/latest.pth"

payload = torch.load(input_path, map_location='cuda')
torch.save(payload['model_state'], output_path)