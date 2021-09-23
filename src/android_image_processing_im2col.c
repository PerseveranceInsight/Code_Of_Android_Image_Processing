#include <stdio.h>

#include "android_arm_util.h"
#include "android_image_processing_im2col.h"

#define IM2COL_GET_PIXEL_LOG LOG_VERBOSE


uint8_t im2col_get_pixel_uint8_t(uint8_t const *in_img,
                                 const int height,
                                 const int width,
                                 const int channels,
                                 const int row,
                                 const int col,
                                 const int channel,
                                 const int pad)
{
    // FUNC_ENTRANCE_LOG;
    uint8_t pixel_content = 0;
    int row_with_padding = 0;
    int col_with_padding = 0;
    // ree_printf(IM2COL_GET_PIXEL_LOG, "%s\nheight %d\nwidth %d\nchannels %d\nrow %d\ncol %d\nchannel %d\npad %d\n", __func__,
    //                                                                                                             height,
    //                                                                                                             width,
    //                                                                                                             channels,
    //                                                                                                             row,
    //                                                                                                             col,
    //                                                                                                             channel,
    //                                                                                                             pad);
    row_with_padding = row - pad;
    col_with_padding = col - pad;
    // ree_log(LOG_DEBUG, "%s row_with_padding %d col_with_padding %d", __func__,
    //                                                                 row_with_padding,
    //                                                                 col_with_padding);
    // ree_printf(LOG_DEBUG, "%s %d\n", __func__, col_with_padding + width*(row_with_padding + height*channel));
    if ((row_with_padding < 0) || (col_with_padding < 0) ||
        (row_with_padding >= height) || (col_with_padding >= width)) 
    {
        pixel_content = 0;
        goto EXIT_IM2COL_GET_PIXEL_UINT8_T;
    }
    pixel_content = in_img[col_with_padding + width*(row_with_padding + height*channel)];
EXIT_IM2COL_GET_PIXEL_UINT8_T:
    // FUNC_EXIT_LOG;
    return pixel_content;
}

void im2col_cpu_uint8_t(uint8_t const *in_img,
                        const int channels,
                        const int height,
                        const int width,
                        const int kernel_size,
                        const int stride,
                        const int pad,
                        uint8_t *in_img_col)
{
    FUNC_ENTRANCE_LOG;
    int c_col_in = 0, h_col_in = 0, w_col_in = 0;
    int col_ind = 0;
    int channel_im = 0;
    int im_row = 0;
    int im_col = 0;
    int width_offset = 0;
    int height_offset = 0;
    int height_col = (height + 2*pad - kernel_size) / stride + 1;
    int width_col = (width + 2*pad - kernel_size) / stride + 1;
    int channel_col = channels * kernel_size * kernel_size;
    ree_log(LOG_DEBUG, "%s height_col %d", __func__, height_col);
    ree_log(LOG_DEBUG, "%s width_col %d", __func__, width_col);
    ree_log(LOG_DEBUG, "%s channel_col %d", __func__, channel_col);

    for (c_col_in = 0; c_col_in < channel_col; c_col_in++)
    {
        width_offset = c_col_in % kernel_size;
        height_offset = (c_col_in / kernel_size) % kernel_size;
        channel_im = (c_col_in / kernel_size)/kernel_size;
        // ree_printf(LOG_DEBUG, "%s width_offset %d height_offset %d channel_im %d\n", __func__, width_offset, height_offset, channel_im);
        for (h_col_in = 0; h_col_in < height_col; h_col_in++)
        {
            for (w_col_in = 0; w_col_in < width_col; w_col_in++)
            {
                im_row = height_offset + h_col_in * stride;
                im_col = width_offset + w_col_in * stride;
                col_ind = (c_col_in * height_col + h_col_in) * width_col + w_col_in; 
                // ree_printf(LOG_DEBUG, "%s im_row %d %d %d\n", __func__, im_row, im_col, col_ind);
                in_img_col[col_ind] = im2col_get_pixel_uint8_t(in_img, height, width, channels,
                                                               im_row, im_col, channel_im, pad);
            }
        }
    }
    FUNC_EXIT_LOG;
}
