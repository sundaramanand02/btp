import cv2

video_path = 'original_video.mp4'
cap = cv2.VideoCapture(video_path)
 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

new_width = width//2
new_height = height//2
 
fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
 
output_path_avi = 'resolution_reduction.avi'
out_avi = cv2.VideoWriter(output_path_avi, fourcc_avi, 30, (new_width, new_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (new_width, new_height))

    out_avi.write(frame)
 
cap.release()
out_avi.release()
cv2.destroyAllWindows()
