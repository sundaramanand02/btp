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


def quantize_coefficients(coefficients, quantization_matrix):
    M, N = coefficients.shape
    m, n = quantization_matrix.shape 
    reshaped_coefficients = coefficients.reshape((M // m, m, N // n, n))
 
    quantized_coefficients = np.round(reshaped_coefficients / quantization_matrix)
 
    return quantized_coefficients.reshape((M, N))


def dequantize_coefficients(quantized_coefficients, quantization_matrix):
    return quantized_coefficients * quantization_matrix
 
video_path = 'output_video_with_motion_compensation.avi'
cap = cv2.VideoCapture(video_path)
 
if not cap.isOpened():
    raise ValueError("Error: Could not open video file")
 
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)   
 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'output_video_with_quantization.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    gray_frame = custom_cvt_color(frame)
 
    dct_frame = fft_dct(np.float32(gray_frame))
 
    idct_frame = idct(idct(dct_frame.T, norm='ortho').T, norm='ortho')
 
    idct_frame = np.uint8(idct_frame)
 
    quantized_frame = quantize_coefficients(idct_frame, quantization_matrix)
 
    dequantized_frame = dequantize_coefficients(quantized_frame, quantization_matrix)
 
    out.write(dequantized_frame)
 
cap.release()
out.release()
cv2.destroyAllWindows()
