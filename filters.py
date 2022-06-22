import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

from scipy.ndimage import convolve
from tqdm import tqdm
from profiler import time_this, cumulative_profiler
from pydicom import dcmread
from typing import Final

DATASET_CONST: Final[str] = 'Shoulder'

#@time_this
def create_frames_list(root='./data'):
    img_wildcard = os.path.join(root, f'{DATASET_CONST}', '*')
    img_path = glob.glob(img_wildcard)
    return img_path

#@time_this
def apply_gaussian_blur(img, kernel_size=5, sigma=1):
    kernel_size = kernel_size // 2
    x, y = np.mgrid[-kernel_size:kernel_size + 1, -kernel_size:kernel_size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    img_blur = convolve(img, g)
    return img_blur

#@time_this
def apply_unsharp_mask(img, blurred, amount=10):
    img_sharpen = img * float(amount + 1) - blurred * float(amount)
    img_sharpen = np.maximum(img_sharpen, np.zeros_like(img_sharpen))
    img_sharpen = np.minimum(img_sharpen, np.full_like(img_sharpen, fill_value=255))
    img_sharpen = img_sharpen.astype(np.uint8)
    return img_sharpen

#@time_this
def compute_sobel_gradients(img):
    img = img / 255.0
    horizontal_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    vertical_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    dx = convolve(img, horizontal_kernel)
    dy = convolve(img, vertical_kernel)
    grad = np.hypot(dx, dy)
    grad = (grad / grad.max() * 255.0).astype(np.uint8)
    theta = np.arctan2(dy, dx)
    return grad, theta

#@time_this
#@numba.jit(parallel=True, nopython=True)
def apply_non_max_suppression(img, theta, first_neighbour_intensity=255, second_neighbour_intensity=255):
    img_nms = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    angle_degree = theta * 180.0 / np.pi
    angle_degree = np.where(angle_degree < 0, angle_degree + 180, angle_degree)
    
    # 3x3 kernel
    for row in range(1, img.shape[0] - 1):
        for column in range(1, img.shape[1] - 1):
            
            # reset the max intensities
            first_neighbour_intensity = 255
            second_neighbour_intensity = 255
            
            # identify the edge direction based on the angle value (can be: -, |, \, /)
            # angle 0
            if 0 <= angle_degree[row, column] < 22.5 or 157.5 <= angle_degree[row, column] <= 180:
                first_neighbour_intensity = img[row, column + 1]
                second_neighbour_intensity = img[row, column - 1]
            # angle 45
            elif 22.5 <= angle_degree[row, column] < 67.5:
                first_neighbour_intensity = img[row + 1, column - 1]
                second_neighbour_intensity = img[row - 1, column + 1]
            # angle 90
            elif 67.5 <= angle_degree[row, column] < 112.5:
                first_neighbour_intensity = img[row + 1, column]
                second_neighbour_intensity = img[row - 1, column]
            # angle 135
            elif 112.5 <= angle_degree[row, column] < 157.5:
                first_neighbour_intensity = img[row - 1, column - 1]
                second_neighbour_intensity = img[row + 1, column + 1]

            # check if the pixel in the same direction has a higher intensity than the pixel that is currently processed
            if img[row, column] >= first_neighbour_intensity and img[row, column] >= second_neighbour_intensity:
                img_nms[row, column] = img[row, column]
            else:
                img_nms[row, column] = 0
    return img_nms

#@time_this
def apply_otsu_binarization(img):
    histogram, bins = np.histogram(img, np.array(range(0, 256)))
    intensities = np.arange(255)
    threshold = -1
    inter_class_variance = -1
    for pixel in bins[1:-1]:
        w1 = np.sum(histogram[:pixel])
        w2 = np.sum(histogram[pixel:])
        m1 = np.sum(intensities[:pixel] * histogram[:pixel]) / float(w1)
        m2 = np.sum(intensities[pixel:] * histogram[pixel:]) / float(w2)
        current_inter_class_variance = w1 * w2 * (m1 - m2) ** 2
        if current_inter_class_variance > inter_class_variance:
            threshold = pixel
            inter_class_variance = current_inter_class_variance
    img_th = np.zeros_like(img)
    img_th[img > threshold] = 255
    return img_th, threshold

#@time_this
def apply_hysteresis(img_hyster, weak, strong=255):
    for row in range(1, img_hyster.shape[0] - 1):
        for col in range(1, img_hyster.shape[1] - 1):
            if img_hyster[row, col] == weak:
                if ((img_hyster[row + 1, col - 1] == strong) or (img_hyster[row + 1, col] == strong) or (img_hyster[row + 1, col + 1] == strong)
                    or (img_hyster[row, col - 1] == strong) or (img_hyster[row, col + 1] == strong)
                    or (img_hyster[row - 1, col - 1] == strong) or (img_hyster[row - 1, col] == strong) or (img_hyster[row - 1, col + 1] == strong)):
                    img_hyster[row, col] = strong
                else:
                    img_hyster[row, col] = 0
    return img_hyster

#@time_this
def apply_colormap(img, rgb=(255, 0, 0)):
    # img_cm = (colormap(img) * 255).astype(np.uint8)[:, :, :3]
    img_cm = np.expand_dims(img, 2)
    img_cm = np.dstack((img_cm, img_cm, img_cm))
    img_cm = np.where(img_cm == (255, 255, 255), rgb[::-1], img_cm) 
    return img_cm

#@time_this
def blend(img, edges, alpha=0.5):
    if np.max(img) >= 255:
        img = img / 255.0
    if np.max(edges) >= 255:
        edges = edges / 255.0
    img_blend = (1.0 - alpha) * img + alpha * edges
    stack = np.hstack((img, img_blend))
    return (stack * 255).astype(np.uint8)

#@time_this
def run_pipeline(filenames):
    for idx, file in tqdm(enumerate(filenames), total=len(filenames)):
        img_gray = dcmread(file).pixel_array.astype(np.uint8)
        img_org = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_blur = apply_gaussian_blur(img_gray)
        img_sharpen = apply_unsharp_mask(img_gray, img_blur)
        img_grad, theta = compute_sobel_gradients(img_sharpen)
        img_nms = apply_non_max_suppression(img_grad, theta)
        img_th, th = apply_otsu_binarization(img_nms)
        img_hyster = apply_hysteresis(img_th, th)
        img_cm = apply_colormap(img_hyster, rgb=(255, 0, 0))
        img_blend = blend(img_org, img_cm, alpha=0.5)
        if not os.path.exists(f'./out/{DATASET_CONST}/cpu'):
            os.makedirs(f'./out/{DATASET_CONST}/cpu')
        cv2.imwrite(f'./out/{DATASET_CONST}/cpu/frame_{idx}.png', img_blend)

#@time_this
@cumulative_profiler(runs=1)
def main():
    filenames = create_frames_list()
    run_pipeline(filenames)
    
if __name__ == '__main__':
    main()