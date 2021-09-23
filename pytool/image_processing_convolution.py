import numpy as np
import time
import logging

def conv_2d(input_matrix, kernel, bias):
    res_matrix = np.copy(input_matrix)
    kernel_flatten = kernel.flatten()
    input_shape = input_matrix.shape
    kernel_shape = kernel.shape
    num_row = input_shape[0]
    num_col = input_shape[1]
    kernel_row = kernel_shape[0]
    kernel_col = kernel_shape[1]
    row_iter_num = np.round(num_row / kernel_row).astype('uint32')
    col_iter_num = np.round(num_col / kernel_col).astype('uint32')
    start_time = time.time_ns()
    for i in range(row_iter_num):
        for j in range(col_iter_num):
            col_vec = input_matrix[i:i+kernel_row, j:j+kernel_col].flatten()
            res_matrix[i, j]=np.sum(np.multiply(col_vec, kernel_flatten))+bias
    end_time = time.time_ns()
    print('res_matrix :\n{0}'.format(res_matrix))
    print('exec time : {0}'.format(end_time - start_time))

def conv_im2col(input_matrix, kernel, bias):
    kernel_flatten = kernel.flatten()

    window_array = im2col(input_matrix, kernel)
    start_time = time.time_ns()
    window_res = np.dot(kernel_flatten, window_array) + bias
    end_time = time.time_ns()
    print('{0} exec time : {1}'.format(conv_im2col.__name__, end_time - start_time))
    print('window_res :\n{0}'.format(window_res))

def im2col(input_matrix, kernel):
    input_shape = input_matrix.shape
    kernel_shape = kernel.shape
    num_row = input_shape[0]
    num_col = input_shape[1]
    kernel_row = kernel_shape[0]
    kernel_col = kernel_shape[1]
    row_iter_num = np.round(num_row / kernel_row).astype('uint32')
    col_iter_num = np.round(num_col / kernel_col).astype('uint32')
    window_array = []
    start_time = time.time_ns()
    for i in range(row_iter_num):
        for j in range(col_iter_num):
            window = input_matrix[i:i+kernel_row,
                                  j:j+kernel_col].flatten()
            window_array.append(window)
    end_time = time.time_ns()
    print('{0} exec time : {1}'.format(im2col.__name__, end_time - start_time))
    return np.asarray(window_array)

def im2col_cpu_py(in_img,
                  channels,
                  high,
                  width,
                  kernel_size,
                  stride,
                  padding):
    def im2col_get_pixel(in_img, channels, high, width, img_row_ind, 
                         img_col_ind, input_channel_ind, padding):
        img_row_ind_wo_padding = img_row_ind - padding
        img_col_ind_wo_padding = img_col_ind - padding
        if (img_row_ind_wo_padding < 0) or (img_col_ind_wo_padding < 0) or (img_row_ind_wo_padding >= high) or (img_col_ind_wo_padding >= width):
            return 0
        else:
            return in_img[img_col_ind_wo_padding + width*(img_row_ind_wo_padding + high*input_channel_ind)]
    '''
        input_matrix = [[0.0, 1.0, 2.0],
                        [3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0]]
        input_matrix with padding = [[0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 2.0, 0.0],
                                     [0.0, 3.0, 4.0, 5.0, 0.0],
                                     [0.0, 6.0, 7.0, 8.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0]]
        output_col = 
                [[0. 0. 0. 0. 0. 0. 1. 2. 0. 3. 4. 5. 0. 6. 7. 8.]    -----             out_channels_ind = 0, width_offset = 0, high_offset = 0, channel_offset = 0
                 [0. 0. 0. 0. 0. 1. 2. 0. 3. 4. 5. 0. 6. 7. 8. 0.]      | 1st channel   out_channels_ind = 1, width_offset = 1, high_offset = 0, channel_offset = 0
                 [0. 0. 1. 2. 0. 3. 4. 5. 0. 6. 7. 8. 0. 0. 0. 0.]      |               out_channels_ind = 2, width_offset = 0, high_offset = 1, channel_offset = 0
                 [0. 1. 2. 0. 3. 4. 5. 0. 6. 7. 8. 0. 0. 0. 0. 0.]]   -----             out_channels_ind = 3, width_offset = 1, high_offset = 1, channel_offset = 0
                  |          output_high * output_width        |
                  w1 w2 w3 w4 w1 w2 w3 w4 w1 w2 w3 w4 w1 w2 w3 w4 
                  h1 h1 h1 h1 h2 h2 h2 h2 h2 h3 h3 h3 h4 h4 h4 h4


    '''
    output_high = (high + 2*padding - kernel_size)//stride + 1
    output_width = (width + 2*padding - kernel_size)//stride + 1
    output_channels = (kernel_size**2)*channels
    output_num = output_high * output_width
    output_col = np.zeros((output_channels, output_num)).flatten()
    for out_channels_ind in range(output_channels):
        width_offset = out_channels_ind % kernel_size
        high_offset = (out_channels_ind // kernel_size) % kernel_size
        input_channel_ind = (out_channels_ind // kernel_size) // kernel_size
        for out_high_ind in range(0, output_high):
            for out_width_ind in range(0, output_width):
               in_img_row_ind = high_offset + (out_high_ind * stride) 
               in_img_col_ind = width_offset + (out_width_ind * stride)
               col_ind = (out_channels_ind * output_high + out_high_ind)*output_width + out_width_ind
               output_col[col_ind] = im2col_get_pixel(in_img, channels, high, width, 
                                                      in_img_row_ind, 
                                                      in_img_col_ind, 
                                                      input_channel_ind, padding)
    return output_col.reshape((output_channels, output_num))

def im2row_cpu_py(in_img,
                  channels,
                  high,
                  width,
                  kernel_size,
                  stride,
                  padding):
    '''
        input_matrix = [[0.0, 1.0, 2.0],
                        [3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0]]
        input_matrix with padding = [[0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 2.0, 0.0],
                                     [0.0, 3.0, 4.0, 5.0, 0.0],
                                     [0.0, 6.0, 7.0, 8.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0]]
        output_row = 
                           1st channel   --> out_channel_ind
                       |<-------------->|
                 -   [[0.0, 0.0, 0.0, 0.0],     w1  h1      ---                                              -                 ---
                 -    [0.0, 0.0, 0.0, 1.0],     w2  h1       ^                                               |   high_offset    |
stride (width)   -    [0.0, 0.0, 1.0, 2.0],     w3  h1       |                                               -                  |
                 -    [0.0, 0.0, 2.0, 0.0],     w4  h1       |                                               |   high_offset    |      stride (high)
                      [0.0, 0.0, 0.0, 3.0],     w1  h2       |                                               -                  |
                      [0.0, 1.0, 3.0, 4.0],     w2  h2       |                                               |   high_offset    |
                      [1.0, 2.0, 4.0, 5.0],     w3  h2       |                                               -                 ---
                      [2.0, 0.0, 5.0, 0.0],     w4  h2       |    output_high * output_width = output_num
                      [0.0, 3.0, 0.0, 6.0],     w1  h3       |
                      [3.0, 4.0, 6.0, 7.0],     w2  h3       |
                      [4.0, 5.0, 7.0, 8.0],     w3  h3       |
                      [5.0, 0.0, 8.0, 0.0],     w4  h3       |
                      [0.0, 6.0, 0.0, 0.0],     w1  h4       |
                      [6.0, 7.0, 0.0, 0.0],     w2  h4       |
                      [7.0, 8.0, 0.0, 0.0],     w3  h4       v
                      [8.0, 0.0, 0.0, 0.0]]     w4  h4      ---  
                         |<----------->|
                           width_offset
    '''
    def im2row_get_pixel(in_img, img_channel_num, img_high, img_width, 
                         row_high_ind, row_width_ind, input_channel_ind, padding):
        row_high_ind_wo_padding = row_high_ind - padding
        row_width_ind_wo_padding = row_width_ind - padding
        if ((row_high_ind_wo_padding<0) or \
            (row_width_ind_wo_padding<0) or \
            (row_high_ind_wo_padding >= img_high) \
            or (row_width_ind_wo_padding >= img_width)):
            return 0
        else:
            return in_img[(row_high_ind_wo_padding + input_channel_ind*img_high)*img_width + row_width_ind_wo_padding]

    output_high = (high + 2*padding - kernel_size)//stride + 1
    output_width = (width + 2*padding - kernel_size)//stride + 1
    output_num = output_high * output_width
    output_channel_num = channels * (kernel_size**2)
    output_row = np.zeros((output_num, output_channel_num)).flatten()
    # print("{0} output_high {1}".format(im2row_cpu_py.__name__, output_high))
    # print("{0} output_width {1}".format(im2row_cpu_py.__name__, output_width))
    # print("{0} output_num {1}".format(im2row_cpu_py.__name__, output_num))
    # print("{0} output_channel_num {1}".format(im2row_cpu_py.__name__, output_channel_num))
    for output_channel_ind in range(output_channel_num):
        width_offset = output_channel_ind % kernel_size
        high_offset = (output_channel_ind // kernel_size)%kernel_size
        input_channel_ind = output_channel_ind // (kernel_size*kernel_size)
        for high_ind in range(output_high):
            for width_ind in range(output_width):
               output_row_ind = (width_ind + high_ind * (output_width))*output_channel_num + output_channel_ind
               output_row[output_row_ind] = im2row_get_pixel(in_img, channels, high, width, 
                                                             high_ind*stride+high_offset, 
                                                             width_ind*stride+width_offset, 
                                                             input_channel_ind, padding)
    return output_row.reshape((output_num, output_channel_num))

def row2im_cpu_py(row_data,
                  channels,
                  high,
                  width,
                  kernel_size,
                  stride,
                  padding):
    def row2im_set_pixel_ind(channels, img_high, img_width,
                             row_high_ind,
                             row_width_ind,
                             input_channel_ind,
                             padding):
        row_high_ind_wo_padding = row_high_ind - padding
        row_width_ind_wo_padding = row_width_ind - padding
        if ((row_high_ind_wo_padding<0) or
            (row_width_ind_wo_padding<0) or
            (row_high_ind_wo_padding>=img_high) or
            (row_width_ind_wo_padding>=img_width)):
            return None
        else:
            return ((row_high_ind_wo_padding + input_channel_ind*img_high)*img_width + row_width_ind_wo_padding)
    output_high = (high + 2*padding - kernel_size)//stride + 1
    output_width = (width + 2*padding - kernel_size)//stride + 1
    output_channels = (kernel_size**2)*channels
    output_img = np.zeros((high, width, channels)).flatten()
    for output_channel_ind in range(output_channels):
        width_offset = output_channel_ind % kernel_size
        high_offset = (output_channel_ind // kernel_size) % kernel_size
        input_channel_ind = output_channel_ind // (kernel_size*kernel_size)
        for output_high_ind in range(output_high):
            for output_width_ind in range(output_width):
                row_data_ind = (output_width_ind + output_high_ind*(output_width))*output_channels + output_channel_ind
                input_pixel_ind = row2im_set_pixel_ind(channels, high, width,
                                                       high_offset + stride*output_high_ind,
                                                       width_offset + stride*output_width_ind,
                                                       input_channel_ind,
                                                       padding)
                if input_pixel_ind is not None:
                    output_img[input_pixel_ind] += row_data[row_data_ind]
    return output_img//(kernel_size**2)

def gemm_py_na(A_tran, B_tran,
               m_dim, n_dim, k_dim,
               alpha, beta,
               mat_A,
               mat_B,
               mat_C):
    def gemm_nn_py_na(m_dim, n_dim, k_dim,
                      alpha,
                      mat_A,
                      mat_B,
                      mat_C):
        for i in range(0, m_dim):
            for k in range(0, k_dim):
                part_a = alpha * mat_A[i*k_dim+k]
                for j in range(0, n_dim):
                    mat_C[i*n_dim+j] = mat_C[i*n_dim+j] + part_a*mat_B[k*n_dim+j]

    def gemm_nt_py_na(m_dim, n_dim, k_dim,
                      alpha,
                      mat_A,
                      mat_B,
                      mat_C):
        for i in range(m_dim):
            for j in range(n_dim):
                micro_kernel_sum = 0
                for k in range(k_dim):
                    micro_kernel_sum += alpha * mat_A[i*k_dim+k]*mat_B[j*k_dim + k]
                mat_C[i*n_dim+j] += micro_kernel_sum

    def gemm_tn_py_na(m_dim, n_dim, k_dim,
                      alpha, 
                      mat_A,
                      mat_B,
                      mat_C):
        for i in range(m_dim):
            for k in range(k_dim):
                part_a = alpha * mat_A[k*m_dim+i]
                for j in range(n_dim):
                    mat_C[i*n_dim + j] = mat_C[i*n_dim + j] + part_a*mat_B[k*n_dim + j]

    def gemm_tt_py_na(m_dim, n_dim, k_dim,
                      alpha,
                      mat_A,
                      mat_B,
                      mat_C):
        for i in range(m_dim):
            for j in range(n_dim):
                micro_kernel_sum = 0
                for k in range(k_dim):
                    micro_kernel_sum += alpha*mat_A[k*m_dim + i]*mat_B[j*k_dim+k]
                mat_C[i*n_dim+j] = micro_kernel_sum
    
    for i in range(0, m_dim):
        for j in range(0, n_dim):
            mat_C[i*n_dim+j] = mat_C[i*n_dim+j]*beta

    if (A_tran is False) and (B_tran is False):
        gemm_nn_py_na(m_dim, n_dim, k_dim,
                      alpha,
                      mat_A, mat_B, mat_C)
    elif (A_tran is True) and (B_tran is False):
        gemm_tn_py_na(m_dim, n_dim, k_dim,
                      alpha, 
                      mat_A, mat_B, mat_C)
    elif (A_tran is False) and (B_tran is True):
        gemm_nt_py_na(m_dim, n_dim, k_dim,
                      alpha, 
                      mat_A, mat_B, mat_C)
    elif (A_tran is True) and (B_tran is True):
        gemm_tt_py_na(m_dim, n_dim, k_dim,
                      alpha, 
                      mat_A, mat_B, mat_C)
    
    return mat_C

def gemm_py_pack(mat_a_tran, mat_b_tran,
                 m_dim, n_dim, k_dim,
                 alpha, beta,
                 mat_a, mat_b, mat_c):
    '''
        m_dim: # of kernels
        k_dim: kernel_size**2
        n_dim: output_hight*output_width
    '''
    def gemm_py_pack_nt(m_dim, n_dim, k_dim,
                        alpha,
                        mat_a, mat_b, mat_c):
        logging.debug(' m_dim : {0}'.format(m_dim))
        logging.debug(' n_dim : {0}'.format(n_dim))
        logging.debug(' k_dim : {0}'.format(k_dim))
        logging.debug(' pack_high : {0}'.format(pack_high))
        logging.debug(' pack_width : {0}'.format(pack_width))
        m_steps = m_dim//pack_high + 1
        n_steps = n_dim//pack_high + 1
        k_steps = k_dim//pack_width + 1
        mat_a_aux = np.zeros((m_steps*k_steps*pack_width*pack_high))
        mat_b_aux = np.zeros((n_steps*k_steps*pack_width*pack_high))
        mat_a_aux[0:m_dim*k_dim] = mat_a[0:m_dim*k_dim]
        mat_b_aux[0:n_dim*k_dim] = mat_b[0:n_dim*k_dim]
        remain_k_lane = pack_width - k_steps*pack_width + k_dim
        logging.debug(' m_steps : {0}'.format(m_steps))
        logging.debug(' n_steps : {0}'.format(n_steps))
        logging.debug(' k_steps : {0}'.format(k_steps))
        logging.debug(' remain_k_lane : {0}'.format(remain_k_lane))
        for m_step in range(m_steps):
            logging.debug(' m_step : {0}'.format(m_step))
            m_offset = m_step * pack_high
            logging.debug(' m_offset : {0}'.format(m_offset))
            for k_step in range(k_steps-1):
                k_offset = k_step * pack_width
                vec_a_aux0 = mat_a_aux[m_offset*k_dim+k_offset:m_offset*k_dim+k_offset+pack_width]*alpha
                vec_a_aux1 = mat_a_aux[(m_offset+1)*k_dim+k_offset:(m_offset+1)*k_dim+k_offset+pack_width]*alpha
                vec_a_aux2 = mat_a_aux[(m_offset+2)*k_dim+k_offset:(m_offset+2)*k_dim+k_offset+pack_width]*alpha
                vec_a_aux3 = mat_a_aux[(m_offset+3)*k_dim+k_offset:(m_offset+3)*k_dim+k_offset+pack_width]*alpha
                logging.debug(' vec_a_aux0 : {0}'.format(vec_a_aux0))
                logging.debug(' vec_a_aux1 : {0}'.format(vec_a_aux1))
                logging.debug(' vec_a_aux2 : {0}'.format(vec_a_aux2))
                logging.debug(' vec_a_aux3 : {0}'.format(vec_a_aux3))
                for n_step in range(n_steps):
                    n_offset = n_step * pack_high
                    vec_b_aux0 = mat_b_aux[n_offset*k_dim+k_offset:n_offset*k_dim+k_offset+pack_width]
                    vec_b_aux1 = mat_b_aux[(n_offset+1)*k_dim+k_offset:(n_offset+1)*k_dim+k_offset+pack_width]
                    vec_b_aux2 = mat_b_aux[(n_offset+2)*k_dim+k_offset:(n_offset+2)*k_dim+k_offset+pack_width]
                    vec_b_aux3 = mat_b_aux[(n_offset+3)*k_dim+k_offset:(n_offset+3)*k_dim+k_offset+pack_width]
                    logging.debug(' vec_b_aux0 : {0}'.format(vec_b_aux0))
                    logging.debug(' vec_b_aux1 : {0}'.format(vec_b_aux1))
                    logging.debug(' vec_b_aux2 : {0}'.format(vec_b_aux2))
                    logging.debug(' vec_b_aux3 : {0}'.format(vec_b_aux3))
                    sca_c_aux00 = np.dot(vec_a_aux0, vec_b_aux0)
                    sca_c_aux01 = np.dot(vec_a_aux0, vec_b_aux1)
                    sca_c_aux02 = np.dot(vec_a_aux0, vec_b_aux2)
                    sca_c_aux03 = np.dot(vec_a_aux0, vec_b_aux3)
                    sca_c_aux10 = np.dot(vec_a_aux1, vec_b_aux0)
                    sca_c_aux11 = np.dot(vec_a_aux1, vec_b_aux1)
                    sca_c_aux12 = np.dot(vec_a_aux1, vec_b_aux2)
                    sca_c_aux13 = np.dot(vec_a_aux1, vec_b_aux3)
                    sca_c_aux20 = np.dot(vec_a_aux2, vec_b_aux0)
                    sca_c_aux21 = np.dot(vec_a_aux2, vec_b_aux1)
                    sca_c_aux22 = np.dot(vec_a_aux2, vec_b_aux2)
                    sca_c_aux23 = np.dot(vec_a_aux2, vec_b_aux3)
                    sca_c_aux30 = np.dot(vec_a_aux3, vec_b_aux0)
                    sca_c_aux31 = np.dot(vec_a_aux3, vec_b_aux1)
                    sca_c_aux32 = np.dot(vec_a_aux3, vec_b_aux2)
                    sca_c_aux33 = np.dot(vec_a_aux3, vec_b_aux3)
                    mat_c[m_offset*n_dim+n_offset:m_offset*n_dim+n_offset+pack_high] += np.array([sca_c_aux00, sca_c_aux01, sca_c_aux02, sca_c_aux03])
                    mat_c[(m_offset+1)*n_dim+n_offset:(m_offset+1)*n_dim+n_offset+pack_high] += np.array([sca_c_aux10, sca_c_aux11, sca_c_aux12, sca_c_aux13])
                    mat_c[(m_offset+2)*n_dim+n_offset:(m_offset+2)*n_dim+n_offset+pack_high] += np.array([sca_c_aux20, sca_c_aux21, sca_c_aux22, sca_c_aux23])
                    mat_c[(m_offset+3)*n_dim+n_offset:(m_offset+3)*n_dim+n_offset+pack_high] += np.array([sca_c_aux30, sca_c_aux31, sca_c_aux32, sca_c_aux33])
            k_offset = (k_steps-1)*pack_width
            vec_a_aux0 = mat_a_aux[m_offset*k_dim+k_offset:m_offset*k_dim+k_offset+remain_k_lane]*alpha
            vec_a_aux1 = mat_a_aux[(m_offset+1)*k_dim+k_offset:(m_offset+1)*k_dim+k_offset+remain_k_lane]*alpha
            vec_a_aux2 = mat_a_aux[(m_offset+2)*k_dim+k_offset:(m_offset+2)*k_dim+k_offset+remain_k_lane]*alpha
            vec_a_aux3 = mat_a_aux[(m_offset+3)*k_dim+k_offset:(m_offset+3)*k_dim+k_offset+remain_k_lane]*alpha
            logging.debug(' vec_a_aux0 : {0}'.format(vec_a_aux0))
            logging.debug(' vec_a_aux1 : {0}'.format(vec_a_aux1))
            logging.debug(' vec_a_aux2 : {0}'.format(vec_a_aux2))
            logging.debug(' vec_a_aux3 : {0}'.format(vec_a_aux3))
            for n_step in range(n_steps):
                n_offset = n_step * pack_high
                vec_b_aux0 = mat_b_aux[n_offset*k_dim+k_offset:n_offset*k_dim+k_offset+remain_k_lane]
                vec_b_aux1 = mat_b_aux[(n_offset+1)*k_dim+k_offset:(n_offset+1)*k_dim+k_offset+remain_k_lane]
                vec_b_aux2 = mat_b_aux[(n_offset+2)*k_dim+k_offset:(n_offset+2)*k_dim+k_offset+remain_k_lane]
                vec_b_aux3 = mat_b_aux[(n_offset+3)*k_dim+k_offset:(n_offset+3)*k_dim+k_offset+remain_k_lane]
                sca_c_aux00 = np.dot(vec_a_aux0, vec_b_aux0)
                sca_c_aux01 = np.dot(vec_a_aux0, vec_b_aux1)
                sca_c_aux02 = np.dot(vec_a_aux0, vec_b_aux2)
                sca_c_aux03 = np.dot(vec_a_aux0, vec_b_aux3)
                sca_c_aux10 = np.dot(vec_a_aux1, vec_b_aux0)
                sca_c_aux11 = np.dot(vec_a_aux1, vec_b_aux1)
                sca_c_aux12 = np.dot(vec_a_aux1, vec_b_aux2)
                sca_c_aux13 = np.dot(vec_a_aux1, vec_b_aux3)
                sca_c_aux20 = np.dot(vec_a_aux2, vec_b_aux0)
                sca_c_aux21 = np.dot(vec_a_aux2, vec_b_aux1)
                sca_c_aux22 = np.dot(vec_a_aux2, vec_b_aux2)
                sca_c_aux23 = np.dot(vec_a_aux2, vec_b_aux3)
                sca_c_aux30 = np.dot(vec_a_aux3, vec_b_aux0)
                sca_c_aux31 = np.dot(vec_a_aux3, vec_b_aux1)
                sca_c_aux32 = np.dot(vec_a_aux3, vec_b_aux2)
                sca_c_aux33 = np.dot(vec_a_aux3, vec_b_aux3)
                mat_c_aux[m_offset*n_dim+n_offset:m_offset*n_dim+n_offset+pack_high] += np.array([sca_c_aux00, sca_c_aux01, sca_c_aux02, sca_c_aux03])
                mat_c_aux[(m_offset+1)*n_dim+n_offset:(m_offset+1)*n_dim+n_offset+pack_high] += np.array([sca_c_aux10, sca_c_aux11, sca_c_aux12, sca_c_aux13])
                mat_c_aux[(m_offset+2)*n_dim+n_offset:(m_offset+2)*n_dim+n_offset+pack_high] += np.array([sca_c_aux20, sca_c_aux21, sca_c_aux22, sca_c_aux23])
                mat_c_aux[(m_offset+3)*n_dim+n_offset:(m_offset+3)*n_dim+n_offset+pack_high] += np.array([sca_c_aux30, sca_c_aux32, sca_c_aux32, sca_c_aux33])

                
    def gemm_py_pack_tn(m_dim, n_dim, k_dim,
                        alpha,
                        mat_a, mat_b, mat_c):
        logging.debug(' m_dim : {0}'.format(m_dim))
        logging.debug(' n_dim : {0}'.format(n_dim))
        logging.debug(' k_dim : {0}'.format(k_dim))

    pack_high = np.int(4)
    pack_width = np.int(4)
    n_steps = n_dim//pack_width+1
    m_steps = m_dim//pack_high+1
    remain_lanes = pack_width - (n_steps*pack_width - n_dim)
    logging.info(' n_steps : {0}'.format(n_steps))
    logging.info(' m_steps : {0}'.format(m_steps))
    logging.info(' remain_lanes : {0}'.format(remain_lanes))
    
    mat_c_aux = np.zeros(n_steps*pack_width*m_steps*pack_width)
    mat_c_aux[0:m_dim*n_dim] = mat_c[0:m_dim*n_dim]
    for m_step in range(m_steps):
        m_offset = m_step * pack_high
        for n_step in range(n_steps-1):
            n_offset = n_step * pack_width
            mat_c_aux[m_offset*n_dim+n_offset:m_offset*n_dim+n_offset+pack_width] += mat_c_aux[m_offset*n_dim+n_offset:m_offset*n_dim+n_offset+pack_width]*beta
            mat_c_aux[(m_offset+1)*n_dim+n_offset:(m_offset+1)*n_dim+n_offset+pack_width] += mat_c_aux[(m_offset+1)*n_dim+n_offset:(m_offset+1)*n_dim+n_offset+pack_width]*beta 
            mat_c_aux[(m_offset+2)*n_dim+n_offset:(m_offset+2)*n_dim+n_offset+pack_width] += mat_c_aux[(m_offset+2)*n_dim+n_offset:(m_offset+2)*n_dim+n_offset+pack_width]*beta
            mat_c_aux[(m_offset+3)*n_dim+n_offset:(m_offset+3)*n_dim+n_offset+pack_width] += mat_c_aux[(m_offset+3)*n_dim+n_offset:(m_offset+3)*n_dim+n_offset+pack_width]*beta
        n_offset = (n_steps-1) * pack_width
        mat_c_aux[m_offset*n_dim+n_offset:m_offset*n_dim+n_offset+remain_lanes] += mat_c_aux[m_offset*n_dim+n_offset:m_offset*n_dim+n_offset+remain_lanes]*beta
        mat_c_aux[(m_offset+1)*n_dim+n_offset:(m_offset+1)*n_dim+n_offset+remain_lanes] += mat_c_aux[(m_offset+1)*n_dim+n_offset:(m_offset+1)*n_dim+n_offset+remain_lanes]*beta
        mat_c_aux[(m_offset+2)*n_dim+n_offset:(m_offset+2)*n_dim+n_offset+remain_lanes] += mat_c_aux[(m_offset+2)*n_dim+n_offset:(m_offset+2)*n_dim+n_offset+remain_lanes]*beta
        mat_c_aux[(m_offset+3)*n_dim+n_offset:(m_offset+3)*n_dim+n_offset+remain_lanes] += mat_c_aux[(m_offset+3)*n_dim+n_offset:(m_offset+3)*n_dim+n_offset+remain_lanes]*beta

    if (mat_a_tran is False) and (mat_b_tran is False):
        logging.warning(' Not support yet NN')
    elif (mat_a_tran is True) and (mat_b_tran is False):
        gemm_py_pack_tn(m_dim, n_dim, k_dim,
                        alpha,
                        mat_a, mat_b, mat_c_aux)
    elif (mat_a_tran is False) and (mat_b_tran is True):
        gemm_py_pack_nt(m_dim, n_dim, k_dim,
                        alpha,
                        mat_a, mat_b, mat_c_aux)
    elif (mat_a_tran is True) and (mat_b_tran is True):
        logging.warning(' Not support yet TT')

    logging.info(' mat_c :\n{0}'.format(mat_c_aux[0:m_dim*n_dim].reshape(m_dim, n_dim).astype('int32')))
    return mat_c_aux[0:m_dim*n_dim]
      
                
if __name__ == '__main__':
    case = 4
    if case == 0:
        input_matrix = np.array([[3.0, 9.0, 0.0],
                                 [2.0, 8.0, 1.0],
                                 [1.0, 4.0, 8.0]])
        kernel = np.array([[8.0, 9.0],
                           [4.0, 4.0]])
        conv_2d(input_matrix,
                kernel, 0.06)
        conv_im2col(input_matrix, kernel, 0.06)
    elif case == 1:
        input_high = 3
        input_width = 3
        input_channels = 2
        kernel_size = 2
        padding = 1
        '''
            input_matrix = [[0.0, 1.0, 2.0],
                            [3.0, 4.0, 5.0],
                            [6.0, 7.0, 8.0]]
            input_matrix with padding = [[0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 1.0, 2.0, 0.0],
                                         [0.0, 3.0, 4.0, 5.0, 0.0],
                                         [0.0, 6.0, 7.0, 8.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0]]
        '''
        input_matrix = np.array([[[0.0, 1.0, 2.0],
                                  [3.0, 4.0, 5.0],
                                  [6.0, 7.0, 8.0]],
                                 [[ 9.0, 10.0, 11.0],
                                  [12.0, 13.0, 14.0],
                                  [15.0, 16.0, 18.0]]])
        output_col = im2col_cpu_py(input_matrix.flatten(),
                                   input_channels,
                                   input_high,
                                   input_width,
                                   kernel_size,
                                   stride = 1,
                                   padding = 1)
        print('output_col :\n{0}'.format(output_col))
    elif case == 2:
        input_high = 3
        input_width = 3
        input_channels = 2
        kernel_size = 2
        padding = 1
        input_matrix = np.array([[[0.0, 1.0, 2.0],
                                  [3.0, 4.0, 5.0],
                                  [6.0, 7.0, 8.0]],
                                 [[ 9.0, 10.0, 11.0],
                                  [12.0, 13.0, 14.0],
                                  [15.0, 16.0, 18.0]]])

        output_row = im2row_cpu_py(input_matrix.flatten(),
                                   input_channels,
                                   input_high,
                                   input_width,
                                   kernel_size,
                                   stride = 1,
                                   padding = padding)
        print('output_row :\n{0}'.format(output_row))
        output_img = row2im_cpu_py(output_row.flatten(),
                                   input_channels,
                                   input_high,
                                   input_width,
                                   kernel_size,
                                   stride = 1,
                                   padding = padding)
        print('output_img : {0}'.format(output_img))
    elif case == 3:
        logging.basicConfig(level=logging.DEBUG)
        mat_a = np.array([[ 3.0, 4.0, 5.0],
                          [ 1.0,10.0, 6.0],
                          [12.0, 9.0, 7.0]])
        mat_b = np.array([[ 2.0, 4.0,12.0],
                          [21.0,45.0,21.0],
                          [32.0,12.0, 4.0]])
        mat_c = np.zeros((3,3))
        print('mat_c :\n{0}'.format(np.matmul(mat_a, mat_b)))
        mat_c = gemm_py_na(True, True,
                           m_dim = 3, n_dim = 3, k_dim = 3,
                           alpha = 1, beta = 1,
                           mat_A = mat_a.transpose().flatten(),
                           mat_B = mat_b.transpose().flatten(),
                           mat_C = mat_c.flatten())
        print('mat_c :\n{0}'.format(mat_c))
        mat_cc = np.zeros((3,3))
        output_mat2 = gemm_py_pack(False, True,
                                   3, 3, 3,
                                   1, 1,
                                   mat_a.flatten(),
                                   mat_b.transpose().flatten(),
                                   mat_cc.flatten())
        print('output_mat2 : {0}'.format(output_mat2))
    elif case == 4:
        np.random.seed(0)
        logging.basicConfig(level=logging.INFO)
        input_high = 7
        input_width = 7
        input_channels = 1
        kernel_size = 3
        kernel_num = 3
        padding = 1
        input_features = np.random.rand(input_high, input_width)*255
        input_features = input_features.astype('int32')
        logging.debug(' input_features :\n{0}'.format(input_features))
        input_kernels = np.random.rand(kernel_num, kernel_size, kernel_size)*5
        input_kernels = input_kernels.astype('int32')
        logging.info(' input_kernels :\n{0}'.format(input_kernels))
        input_features.tofile('./input_featues.bin')
        input_kernels.tofile('./input_kernels.bin')
        row_features = im2row_cpu_py(input_features.flatten(),
                                     input_channels,
                                     input_high,
                                     input_width,
                                     kernel_size,
                                     stride = 1,
                                     padding = padding)
        row_kernels = input_kernels.flatten().reshape(kernel_num, kernel_size**2)
        logging.info(' row_features :\n{0}'.format(row_features))
        # print('input_kernels :\n{0}'.format(row_kernels))
        output_result = np.matmul(row_kernels, row_features.transpose())
        # print('output_result :\n{0}'.format(output_result.astype('int32')))
        # print('shape of row_features : {0}'.format(row_features.shape))
        # print('shape of row_kernels: {0}'.format(row_kernels.shape))
        output_mat = np.zeros((3, 49))
        output_mat = gemm_py_na(False, True,
                                kernel_num, 49, kernel_size**2,
                                1, 0,
                                row_kernels.flatten(),
                                row_features.flatten(),
                                output_mat.flatten())
        print('output_mat :\n{0}'.format(output_mat.reshape((3,49)).astype('int32')))
        output_mat2 = np.zeros((3,49)).flatten()
        output_mat2 = gemm_py_pack(False, True,
                                   kernel_num, 49, kernel_size**2,
                                   1, 0,
                                   row_kernels.flatten(),
                                   row_features.flatten(),
                                   output_mat2.flatten())
