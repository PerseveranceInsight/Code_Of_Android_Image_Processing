#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void im2col_cpu_uint32_t(im2col_param_metadata_t *p_im2col_params)
{
    IM2COL_FUNC_ENTRANCE_LOG;
    int in_channels = p_im2col_params->in_channels;
    int in_heights = p_im2col_params->in_heights;
    int in_widths = p_im2col_params->in_widths;
    int in_kernel_sizes = p_im2col_params->in_kernel_sizes;
    int in_pad_sizes = p_im2col_params->in_pad_sizes;
    int in_stride_sizes = p_im2col_params->in_stride_sizes;
    int row_ind  = 0;
    int out_img_heights = 0, out_img_widths = 0, out_mat_heights = 0, out_mat_widths = 0;
    int out_mat_heights_offset = 0, out_mat_widths_offset = 0, out_mat_channels_offset = 0;
    int out_ele_num = 0;
    int pack_widths = p_im2col_params->pack_widths;
    int out_pack_heights = p_im2col_params->out_pack_heights;
    uint32_t *in_img = NULL;
    uint32_t *col_features = NULL;
    im2col_subparam_metadata_t subparam = {0};

    if (!p_im2col_params)
    {
        ree_log(LOG_ERROR, "%s occurs error due to p_im2col_params is NULL", __func__);
        goto EXIT_IM2COL_CPU_UINT32;
    }
    
    in_img = (uint32_t*)p_im2col_params->in_img;
    if (!in_img)
    {
        ree_log(LOG_ERROR, "%s occurs error due to in_img is NULL", __func__);
        goto EXIT_IM2COL_CPU_UINT32;
    }
    out_img_widths = (in_widths+2*in_pad_sizes - in_kernel_sizes)/in_stride_sizes + 1;
    out_img_heights = (in_heights+2*in_pad_sizes - in_kernel_sizes)/in_stride_sizes + 1;
    out_mat_widths = out_img_widths * out_img_heights;
    out_mat_heights = in_kernel_sizes*in_kernel_sizes*in_channels;

    out_ele_num = out_mat_widths*out_mat_heights;
    ree_log(LOG_DEBUG, "%s out_img_widths %d", __func__, out_img_widths);
    ree_log(LOG_DEBUG, "%s out_img_heights %d", __func__, out_img_heights);
    ree_log(LOG_DEBUG, "%s out_mat_widths %d", __func__, out_mat_widths);
    ree_log(LOG_DEBUG, "%s out_mat_heights %d", __func__, out_mat_heights);
    ree_log(LOG_DEBUG, "%s out_ele_num %d", __func__, out_ele_num);

    col_features = (uint32_t*)ree_malloc(out_ele_num*sizeof(uint32_t));
    if (!col_features)
    {
        ree_log(LOG_ERROR, "%s allocates col_features error", __func__);
        goto EXIT_IM2COL_CPU_UINT32;
    }
    ree_set(col_features, 0, sizeof(uint32_t)*out_ele_num);

    subparam.in_img = (uint8_t*)in_img;
    subparam.in_heights = in_heights;
    subparam.in_widths = in_widths;
    subparam.in_padding = in_pad_sizes;

    for (int out_mat_row = 0; out_mat_row<out_mat_heights; out_mat_row++)
    {
        out_mat_widths_offset = out_mat_row % in_kernel_sizes;
        out_mat_heights_offset = (out_mat_row/in_kernel_sizes)%in_kernel_sizes;
        out_mat_channels_offset = (out_mat_row/in_kernel_sizes)/in_kernel_sizes;
        subparam.in_channel_ind = out_mat_channels_offset;
        for (int out_img_height = 0; out_img_height<out_img_heights; out_img_height++)
        {
            for (int out_img_width = 0; out_img_width<out_img_widths; out_img_width++)
            {
                subparam.col_data_col_ind = out_mat_widths_offset + out_img_width*in_stride_sizes;
                subparam.col_data_row_ind = out_mat_heights_offset + out_img_height*in_stride_sizes;
                row_ind = out_mat_row*out_mat_widths + out_img_height*out_img_widths + out_img_width;
                // ree_log(LOG_DEBUG, "%d", row_ind);
                col_features[row_ind] = im2col_get_pixel_uint32_t(&subparam);
            }
        }
    }

    p_im2col_params->out_img_heights = out_img_heights;
    p_im2col_params->out_img_widths = out_img_widths;
    p_im2col_params->out_mat_heights = out_mat_heights;
    p_im2col_params->out_mat_widths = out_mat_widths;
    p_im2col_params->out_ele_num = out_ele_num;
    p_im2col_params->col_features = (uint8_t*)col_features;
EXIT_IM2COL_CPU_UINT32:
    IM2COL_FUNC_EXIT_LOG;
}
