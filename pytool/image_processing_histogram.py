import numpy as np
import copy 
import matplotlib.pyplot as plt

def histogram_gen(gray_img_flatten):
    gray_hist = np.zeros((256))
    gray_img_shape = gray_img_flatten.shape
    gray_img_size = gray_img_shape[0]
    for i in range(gray_img_size):
        gray_hist[gray_img_flatten[i]] += 1
    return gray_hist

def histogram_equalization_map(gray_hist, pixel_num):
    gray_hist_map = np.zeros((256)).astype('float32')
    accum_num = 0
    for i in range(256):
        accum_num += gray_hist[i]
        gray_hist_map[i] = accum_num
    cdf_min_ind = np.where(gray_hist_map!=0)[0][0]
    cdf_min = gray_hist[cdf_min_ind]
    for i in range(256):
        accum_num = gray_hist_map[i]
        if accum_num < cdf_min:
            continue
        else:
            gray_hist_map[i] = (accum_num)*255.0/(pixel_num)
    return gray_hist_map.astype('uint8')

def histogram_equalization(gray_img, gray_hist_map, pixel_num):
    res_image = np.zeros((pixel_num)).astype('uint8')
    for i in range(pixel_num):
        res_image[i] = gray_hist_map[gray_img[i]]
    return res_image

def histogram_matching(hist_in_eq_map, hist_ref_eq_map):
    hist_in_match_map = copy.deepcopy(hist_in_eq_map)
    tmp = 0
    for i in range(256):
        s_i = hist_in_eq_map[i]
        for j in range(tmp, 256):
            z_j = hist_ref_eq_map[j]
            if ((z_j - s_i)>0):
                hist_in_match_map[i] = j-1
                tmp = j
                break
    return hist_in_match_map



if __name__ == '__main__':
    running_case = 1
    if running_case == 0:
        data_path = '../data/histogram/ipp_histogram_sample.bin'
        eq_res_path = '../data/histogram/ipp_his_eq_res_py.bin' 
        gray_img = np.fromfile(data_path, dtype='uint8').reshape((512, 512))
        print(gray_img)
        gray_img_flatten = gray_img.flatten()
        gray_hist = histogram_gen(gray_img_flatten).astype('uint32')
        print('Histogram :\n {0}'.format(gray_hist))
        gray_hist_map = histogram_equalization_map(gray_hist, 512*512)
        print('Histogram map :\n {0}'.format(gray_hist_map))
        hist_img = histogram_equalization(gray_img_flatten, gray_hist_map, 512*512)
        hist_img.astype('uint8').tofile(eq_res_path)
    elif running_case == 1:
        in_img_path = '../data/histogram/ipp_histogram_matching_in.bin'
        ref_img_path = '../data/histogram/ipp_histogram_matching_ref.bin'
        in_his_match_res_path_py = '../data/histogram/ipp_his_mat_res_py.bin'
        gray_in_img = np.fromfile(in_img_path, dtype='uint8')
        gray_ref_img = np.fromfile(ref_img_path, dtype='uint8')
        in_hist = histogram_gen(gray_in_img).astype('uint32')
        ref_hist = histogram_gen(gray_ref_img).astype('uint32')
        in_hist_eq_map = histogram_equalization_map(in_hist, 846*723)
        print('in_hist_eq_map: {0}'.format(in_hist_eq_map))
        ref_hist_eq_map = histogram_equalization_map(ref_hist, 908*723)
        print('ref_hist_eq_map: {0}'.format(ref_hist_eq_map))
        hist_in_match_map = histogram_matching(in_hist_eq_map.astype('int32'),
                                               ref_hist_eq_map.astype('int32'))
        print('hist_in_match_map: {0}'.format(hist_in_match_map))
        hist_match_res_img = histogram_equalization(gray_in_img, hist_in_match_map, 846*723)
        hist_match_res_img.astype('uint8').tofile(in_his_match_res_path_py)
