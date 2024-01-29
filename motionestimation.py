import cv2
import numpy as np
 
video_path = 'resolution_reduction.avi' 
cap = cv2.VideoCapture(video_path)
 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'motion_estimation.avi'
out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
 
block_size = 64  # Size of the block for motion estimation   
max_displacement = 8  # Maximum displacement to search within

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    output_frame = frame.copy()

    for y in range(0, gray_frame.shape[0], block_size):
        for x in range(0, gray_frame.shape[1], block_size):
            search_area = gray_frame[y:y + block_size, x:x + block_size]

            template = gray_frame[y:y + block_size, x:x + block_size]

            result = cv2.matchTemplate(search_area, template, cv2.TM_SQDIFF_NORMED)

            min_val, _, min_loc, _ = cv2.minMaxLoc(result)
            dx, dy = min_loc
            cv2.arrowedLine(output_frame, (x + block_size // 2, y + block_size // 2),  (x + dx + block_size // 2, y + dy + block_size // 2), (0, 0, 255), 1)

    out.write(output_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
