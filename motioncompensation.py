import cv2
import numpy as np
 
video_path = 'motion_estimation.avi'  
cap = cv2.VideoCapture(video_path)
 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'motion_compensation.avi'
out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

prev_frame = None
motion_thresh = 30  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray_frame
        continue

    flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mean_motion_horizontal = np.mean(np.abs(flow[:, :, 0]))
    mean_motion_vertical = np.mean(np.abs(flow[:, :, 1]))

    mean_motion = np.mean([mean_motion_horizontal, mean_motion_vertical])

    if mean_motion > motion_thresh:
        shift_matrix = np.float32([[1, 0, int(mean_motion_horizontal)], [0, 1, int(mean_motion_vertical)]])
        frame = cv2.warpAffine(frame, shift_matrix, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    prev_frame = gray_frame

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
