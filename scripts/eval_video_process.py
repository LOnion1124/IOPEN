import cv2
import tqdm

path_to_video = "D:/projects/IOPEN/data/eval/video/downsampled.mp4"
output_dir = "D:/projects/IOPEN/data/eval/frame/"

# H, W = 480, 640

cap = cv2.VideoCapture(path_to_video)
frame_count = 2850

for frame_idx in tqdm.tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break
    
    # resized_frame = cv2.resize(frame, (W, H))
    output_path = f"{output_dir}/frame_{frame_idx:06d}.png"
    # cv2.imwrite(output_path, resized_frame)
    cv2.imwrite(output_path, frame)

cap.release()
print(f"Extracted {frame_count} frames")