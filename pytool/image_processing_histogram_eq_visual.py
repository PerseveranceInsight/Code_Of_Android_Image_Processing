import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    case = 1

    if case == 0:
        hist = 0
    elif case == 1:
        hist_match_in_path = '../data/histogram/ipp_histogram_matching_in.bin'
        hist_match_ref_path = '../data/histogram/ipp_histogram_matching_ref.bin'
        hist_match_res_py_path = '../data/histogram/ipp_his_mat_res_py.bin'
        hist_match_res_c_path = '../data/histogram/ipp_his_mat_res_c.bin'
        hist_match_res_neon_path = '../data/histogram/ipp_his_mat_res_neon.bin' 
        hist_match_in = np.fromfile(hist_match_in_path, dtype='uint8')
        hist_match_ref = np.fromfile(hist_match_ref_path, dtype='uint8')
        hist_match_res_py = np.fromfile(hist_match_res_py_path, dtype='uint8')
        hist_match_res_c = np.fromfile(hist_match_res_c_path, dtype='uint8')
        hist_match_res_neon = np.fromfile(hist_match_res_neon_path, dtype='uint8')
        plt.figure()
        plt.subplot(3, 2, 1)
        plt.title('Histogram matching input')
        hist_match_in = hist_match_in.reshape(846, 723)
        plt.imshow(hist_match_in, cmap='gray', vmin=0, vmax=255)
        plt.subplot(3, 2, 2)
        plt.title('Histogram matching reference')
        hist_match_ref = hist_match_ref.reshape(908, 723)
        plt.imshow(hist_match_ref, cmap='gray', vmin=0, vmax=255)
        plt.subplot(3, 2, 3)
        plt.title('Histogram matching result Py')
        hist_match_res_py = hist_match_res_py.reshape(846, 723)
        plt.imshow(hist_match_res_py, cmap='gray', vmin=0, vmax=255)
        plt.subplot(3, 2, 4)
        plt.title('Histogram matching result C')
        hist_match_res_c = hist_match_res_c.reshape(846, 723)
        plt.imshow(hist_match_res_c, cmap='gray', vmin=0, vmax=255)
        plt.subplot(3, 2, 5)
        plt.title('Histogram matching result NEON')
        hist_match_res_neon = hist_match_res_neon.reshape(846, 723)
        plt.imshow(hist_match_res_neon, cmap='gray', vmin=0, vmax=255)
