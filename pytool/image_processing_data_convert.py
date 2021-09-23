import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_path = '../data/histogram/ipp_histogram_matching_ref.jpeg'
    result_path = '../data/histogram/ipp_histogram_matching_ref.bin'

    ori_img = cv.imread(data_path, cv.IMREAD_GRAYSCALE)
    print('Shape of ori_img : {0}'.format(ori_img.shape))
    ori_img.astype('uint8').tofile(result_path)
