import argparse
import cv2
import os

parser = argparse.ArgumentParser(description="Generate video from evaluation result images.")
parser.add_argument("--input-dir", default="data/eval/result", help="Directory containing input images")
parser.add_argument("--output-path", default="data/eval/video/output.mp4", help="Output video file path")
args = parser.parse_args()

input_dir = args.input_dir
output_path = args.output_path

# Get sorted list of image files
images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

if not images:
    print(f"No .jpg images found in {input_dir}")
    exit(1)

# Read first image to get dimensions
first_image = cv2.imread(os.path.join(input_dir, images[0]))
height, width = first_image.shape[:2]

# Create video writer
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

# Write frames
for img_file in images:
    frame = cv2.imread(os.path.join(input_dir, img_file))
    out.write(frame)

out.release()
print(f"Video saved to {output_path}")
print(f"Video saved to {output_path}")