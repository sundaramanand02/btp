import cv2

def downsample_video(input_path, output_path, scale_factor=0.5): 
    cap = cv2.VideoCapture(input_path)
 
    width = int(cap.get(3))
    height = int(cap.get(4))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)   
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (int(width * scale_factor), int(height * scale_factor)))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
 
        downscaled_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
 
        out.write(downscaled_frame)
 
    cap.release()
    out.release()

    cv2.destroyAllWindows()
 
input_video_path = 'original_video.mp4'
output_video_path = 'video_123.avi'
downsample_video(input_video_path, output_video_path)