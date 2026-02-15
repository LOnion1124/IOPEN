import cv2

path_to_video = ""
output_dir = ""

H, W = 480, 640

cap = cv2.VideoCapture(path_to_video)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    resized_frame = cv2.resize(frame, (W, H))
    output_path = f"{output_dir}/frame_{frame_count:06d}.png"
    cv2.imwrite(output_path, resized_frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames")