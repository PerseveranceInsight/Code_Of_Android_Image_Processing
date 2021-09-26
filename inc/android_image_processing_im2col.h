#ifndef __ANDROID_IMAGE_PROCESSING_IM2COL_H__
#define __ANDROID_IMAGE_PROCESSING_IM2COL_H__
#include <stdint.h>
#include "android_arm_util.h"

#ifdef EN_IM2COL_DEBUG
#define IM2COL_FUNC_ENTRANCE_LOG FUNC_ENTRANCE_LOG
#define IM2COL_FUNC_EXIT_LOG FUNC_EXIT_LOG
#else
#define IM2COL_FUNC_ENTRANCE_LOG do {} while(0)
#define IM2COL_FUNC_EXIT_LOG do {} while(0)
#endif

#define IM2COL_INVALID_IM_IND (-999)

typedef struct im2col_subparam_metadata {
    int in_channel_ind;
    int in_heights;
    int in_widths;
    int in_padding;
    int col_data_row_ind;
    int col_data_col_ind;
    uint8_t *in_img; 
} im2col_subparam_metadata_t;

typedef struct im2col_param_metadata {
    int in_channels;
    int in_heights;
    int in_widths;
    int in_kernel_sizes;
    int in_pad_sizes;
    int in_stride_sizes;
    int out_img_heights;
    int out_img_widths;
    int out_mat_heights;
    int out_mat_widths;
    int out_ele_num;
    int pack_widths;
    int kernel_pack_heights;
    int out_pack_heights;
    uint8_t *in_img;
    uint8_t *col_features;
} im2col_param_metadata_t;

static uint32_t inline im2col_get_pixel_uint32_t(im2col_subparam_metadata_t *p_subparam) 
{
    IM2COL_FUNC_ENTRANCE_LOG;
    int in_heights = p_subparam->in_heights;
    int in_widths = p_subparam->in_widths;
    int in_channel_ind = p_subparam->in_channel_ind;
    int col_data_col_ind = p_subparam->col_data_col_ind;
    int col_data_row_ind = p_subparam->col_data_row_ind;
    int padding = p_subparam->in_padding;
    int col_data_col_ind_wo_padding = 0, col_data_row_ind_wo_padding = 0;
    uint32_t pixel_content = 0;
    uint32_t *p_img = (uint32_t*)p_subparam->in_img;
    col_data_col_ind_wo_padding = col_data_col_ind - padding;
    col_data_row_ind_wo_padding = col_data_row_ind - padding;
    if ((col_data_col_ind_wo_padding < 0) ||
        (col_data_row_ind_wo_padding < 0) ||
        (col_data_col_ind_wo_padding >= in_widths) ||
        (col_data_row_ind_wo_padding >= in_heights))
    {
        pixel_content = 0;
    }
    else
    {
        pixel_content = p_img[(in_channel_ind*in_heights+col_data_row_ind_wo_padding)*in_widths + col_data_col_ind_wo_padding]; 
    }
EXIT_IM2COL_GET_PIXEL_UINT32_T:
    IM2COL_FUNC_EXIT_LOG;
    return pixel_content;
};

uint8_t im2col_get_pixel_uint8_t(uint8_t const *in_img,
                                 const int height,
                                 const int width,
                                 const int channels,
                                 const int row,
                                 const int col,
                                 const int channel,
                                 const int pad);
void im2col_cpu_uint8_t(uint8_t const *in_img,
                        const int channels,
                        const int heights,
                        const int width,
                        const int kernel_size,
                        const int stride,
                        const int pad,
                        uint8_t *in_img_col);
void im2col_cpu_uint32_t(im2col_param_metadata_t *p_im2col_params);
#endif
