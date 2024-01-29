import cv2
import numpy as np
from scipy.fftpack import dct, idct

def custom_cvt_color(frame):
    if len(frame.shape) == 3:
        gray_frame = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        return np.uint8(gray_frame)
    else:
        return frame

def custom_GRAY2BGR(frame):
    return cv2.merge([frame, frame, frame])

def fft_dct(frame):
    return dct(dct(frame.T, norm='ortho').T, norm='ortho')

# def block_dct(frame, block_size=8):
#     M, N = frame.shape
#     dct_result = np.zeros((M, N), dtype=float)

#     for i in range(0, M, block_size):
#         for j in range(0, N, block_size):
#             block = frame[i:i+block_size, j:j+block_size]
#             dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
#             dct_result[i:i+block_size, j:j+block_size] = dct_block

#     return dct_result

# def block_idct(frame, block_size=8):
#     M, N = frame.shape
#     idct_result = np.zeros((M, N), dtype=float)

#     for i in range(0, M, block_size):
#         for j in range(0, N, block_size):
#             block = frame[i:i+block_size, j:j+block_size]
#             idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
#             idct_result[i:i+block_size, j:j+block_size] = idct_block

#     return np.uint8(idct_result)

video_path = 'motion_compensation.avi'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Error: Could not open video file")

width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'dct.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # dct_frame = block_dct(np.float32(gray_frame))
    # idct_frame = block_idct(dct_frame)

    # reconstructed_frame = custom_GRAY2BGR(idct_frame)

    dct_frame = fft_dct(np.float32(gray_frame))

    idct_frame = np.uint8(idct(idct(dct_frame.T, norm='ortho').T, norm='ortho'))

    reconstructed_frame = custom_GRAY2BGR(idct_frame)

    out.write(reconstructed_frame)

cap.release()
out.release()
cv2.destroyAllWindows()


def custom_dct(frame):
    M, N = frame.shape
    dct_result = np.zeros((M, N), dtype=float)

    for u in range(M):
        for v in range(N):
            cu = np.sqrt(2/M) if u == 0 else 1
            cv = np.sqrt(2/N) if v == 0 else 1
            sum_val = 0.0

            for x in range(M):
                for y in range(N):
                    sum_val += frame[x, y] * np.cos((2*x + 1) * u * np.pi / (2 * M)) * np.cos((2*y + 1) * v * np.pi / (2 * N))

            dct_result[u, v] = cu * cv * sum_val

    return dct_result