from cProfile import run
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cpx
import cv2
import numba
import matplotlib.pyplot as plt
import os
import glob

from tqdm import tqdm
from typing import Final
from profiler import time_this, cumulative_profiler
from multiprocessing import Pool
from pydicom import dcmread

DATASET_CONST: Final[str] = 'Head'

#@time_this
@numba.jit(parallel=True, fastmath=True, forceobj=True)
def create_frames_list(path_to_image='./data'):
    img_wildcard = os.path.join(path_to_image, f'{DATASET_CONST}', '*')
    img_path = glob.glob(img_wildcard)
    return img_path

#@time_this
@numba.jit(parallel=True, fastmath=True, forceobj=True)
def apply_gaussian_blur_gpu(img, kernel_size=3, sigma=1):
    kernel_size = int(kernel_size) // 2
    x, y = cp.mgrid[-kernel_size:kernel_size + 1, -kernel_size:kernel_size + 1]
    normal = 1 / (2.0 * cp.pi * sigma**2)
    g =  cp.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    img_blur = cpx.convolve(img, g)
    return img_blur

#@time_this
@numba.jit(parallel=True, fastmath=True, forceobj=True)
def apply_unsharp_mask_gpu(img, blurred, amount=2.0, threshold=0):
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = cp.maximum(sharpened, cp.zeros(sharpened.shape))
    sharpened = cp.minimum(sharpened, 255 * cp.ones(sharpened.shape))
    sharpened = sharpened.round().astype(cp.uint8)
    if threshold > 0:
        low_contrast_mask = cp.absolute(img - blurred) < threshold
        cp.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened

#@time_this
@numba.jit(parallel=True, fastmath=True, forceobj=True)
def compute_sobel_gradients_gpu(img):
    img = img / 255.0
    horizontal_kernel = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], cp.float32)
    vertical_kernel = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], cp.float32)
    dx = cpx.convolve(img, horizontal_kernel)
    dy = cpx.convolve(img, vertical_kernel)
    grad = cp.hypot(dx, dy)
    grad = (grad / grad.max() * 255.0).astype(cp.uint8)
    theta = cp.arctan2(dy, dx)
    return grad.get(), theta.get()

#@time_this
@numba.jit(parallel=True, fastmath=True, nopython=True)
def apply_non_max_suppression_gpu(img, theta, first_neighbour_intensity=255, second_neighbour_intensity=255):
    img_nms = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    angle_degree = theta * 180.0 / np.pi
    angle_degree = np.where(angle_degree < 0, angle_degree + 180, angle_degree)
    
    # 3x3 kernel
    for row in numba.prange(1, img.shape[0] - 1):
        for column in numba.prange(1, img.shape[1] - 1):
            
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
@numba.jit(parallel=True, fastmath=True, forceobj=True)
def apply_otsu_binarization_gpu(img):
    img = cp.asarray(img)
    histogram, bins = cp.histogram(img, cp.array(range(0, 256)))
    intensities = cp.arange(255)
    threshold = -1
    inter_class_variance = -1
    for pixel in bins[1:-1]:
        w1 = cp.sum(histogram[:pixel])
        w2 = cp.sum(histogram[pixel:])
        m1 = cp.sum(intensities[:pixel] * histogram[:pixel]) / float(w1)
        m2 = cp.sum(intensities[pixel:] * histogram[pixel:]) / float(w2)
        current_inter_class_variance = w1 * w2 * (m1 - m2) ** 2
        if current_inter_class_variance > inter_class_variance:
            threshold = pixel
            inter_class_variance = current_inter_class_variance
    img_th = cp.zeros_like(img)
    img_th[img > threshold] = 255
    return img_th.get(), threshold.get()

#@time_this
@numba.jit(parallel=True, fastmath=True, nopython=True)
def apply_hysteresis_gpu(img_hyster, weak, strong=255):
    for row in numba.prange(1, img_hyster.shape[0] - 1):
        for col in numba.prange(1, img_hyster.shape[1] - 1):
            if img_hyster[row, col] == weak:
                if ((img_hyster[row + 1, col - 1] == strong) or (img_hyster[row + 1, col] == strong) or (img_hyster[row + 1, col + 1] == strong)
                    or (img_hyster[row, col - 1] == strong) or (img_hyster[row, col + 1] == strong)
                    or (img_hyster[row - 1, col - 1] == strong) or (img_hyster[row - 1, col] == strong) or (img_hyster[row - 1, col + 1] == strong)):
                    img_hyster[row, col] = strong
                else:
                    img_hyster[row, col] = 0
    return img_hyster

#@time_this
@numba.jit(parallel=True, fastmath=True, forceobj=True)
def apply_colormap_gpu(img, rgb=(255, 0, 0)):
    img_cm = np.expand_dims(img, 2)
    img_cm = np.dstack((img_cm, img_cm, img_cm))
    img_cm = np.where(img_cm == (255, 255, 255), rgb[::-1], img_cm) 
    return img_cm

#@time_this
@numba.jit(parallel=True, fastmath=True, forceobj=True)
def blend(img, edges, alpha=0.5):
    if np.max(img) >= 255:
        img = img / 255.0
    if np.max(edges) >= 255:
        edges = edges / 255.0
    img_blend = (1.0 - alpha) * img + alpha * edges
    stack = np.hstack((img, img_blend))
    return (stack * 255).astype(np.uint8)
    
#@time_this
def run_pipeline_gpu(filename):
    # for idx, file in tqdm(enumerate(filenames)):
    #     img_org = cv2.imread(file)
    #     img_gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
    #     img_gpu = cp.asarray(img_gray)
    #     img_blur_gpu = apply_gaussian_blur_gpu(img_gpu)
    #     img_sharpen_gpu = apply_unsharp_mask_gpu(img_gpu, img_blur_gpu)
    #     img_grad_gpu, theta = compute_sobel_gradients_gpu(img_sharpen_gpu)
    #     img_grad_gpu = img_grad_gpu.get()
    #     theta = theta.get()
    #     img_nms_gpu = apply_non_max_suppression_gpu(img_grad_gpu, theta)
    #     img_th_gpu, th = apply_otsu_binarization_gpu(img_nms_gpu)
    #     img_th_gpu = img_th_gpu.get()
    #     th = th.get()
    #     img_hyster_gpu = apply_hysteresis_gpu(img_th_gpu, th)
    #     img_cm_gpu = apply_colormap_gpu(img_hyster_gpu, rgb=(255, 0, 0))
    #     img_blend_gpu = blend(img_org, img_cm_gpu, alpha=0.5)
    #     cv2.imwrite(f'./out/vid1/gpu/frame_{idx}.png', img_blend_gpu)
    
    # img_org = cv2.imread(filename)
    # img_gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)
    img_gray = dcmread(filename).pixel_array.astype(np.uint8)
    img_org = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_gpu = cp.asarray(img_gray)
    img_blur_gpu = apply_gaussian_blur_gpu(img_gpu)
    img_sharpen_gpu = apply_unsharp_mask_gpu(img_gpu, img_blur_gpu)
    img_grad_gpu, theta = compute_sobel_gradients_gpu(img_sharpen_gpu)
    img_nms_gpu = apply_non_max_suppression_gpu(img_grad_gpu, theta)
    img_th_gpu, th = apply_otsu_binarization_gpu(img_nms_gpu)
    img_hyster_gpu = apply_hysteresis_gpu(img_th_gpu, th)
    img_cm_gpu = apply_colormap_gpu(img_hyster_gpu, rgb=(255, 0, 0))
    img_blend_gpu = blend(img_org, img_cm_gpu, alpha=0.5)
    return img_blend_gpu

#@time_this
@cumulative_profiler(runs=1)
def main():
    filenames = create_frames_list()
    if not os.path.exists(f'./out/{DATASET_CONST}/gpu'):
        os.makedirs(f'./out/{DATASET_CONST}/gpu')
    with Pool(processes=4) as pool:
        for idx, img in enumerate(tqdm(pool.imap(run_pipeline_gpu, filenames), total=len(filenames))):
            cv2.imwrite(f'./out/{DATASET_CONST}/gpu/frame_{idx}.png', img)
    
if __name__ == '__main__':
    main()