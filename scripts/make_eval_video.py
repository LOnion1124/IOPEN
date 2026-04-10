import cv2
import os

input_dir = "/home/luyizhi/IOPEN/data/eval/result"
output_dir = "/home/luyizhi/IOPEN/data/eval/video"

# Get sorted list of image files
images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

# Read first image to get dimensions
first_image = cv2.imread(os.path.join(input_dir, images[0]))
height, width = first_image.shape[:2]

# Create video writer
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'output.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

# Write frames
for img_file in images:
    frame = cv2.imread(os.path.join(input_dir, img_file))
    out.write(frame)

out.release()
print(f"Video saved to {output_path}")