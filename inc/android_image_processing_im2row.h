#ifndef __ANDROID_IMAGE_PROCESSING_IM2ROW_H__
#define __ANDROID_IMAGE_PROCESSING_IM2ROW_H__
#include <stdio.h>

#include "android_arm_util.h"

#ifdef EN_IM2ROW_DEBUG
#define IM2ROW_FUNC_ENTRANCE_LOG FUNC_ENTRANCE_LOG
#define IM2ROW_FUNC_EXIT_LOG FUNC_EXIT_LOG
#else
#define IM2ROW_FUNC_ENTRANCE_LOG do {} while(0)
#define IM2ROW_FUNC_EXIT_LOG do {} while(0)
#endif

#define IM2ROW_INVALID_IM_IND (-999)

static uint8_t inline im2row_get_pixel_uint8_t(uint8_t const *in_img,
                                               const int in_highs,
                                               const int in_widths,
                                               const int row_data_high,
                                               const int row_data_width,
                                               const int input_channel_ind,
                                               const int padding)
{
    IM2ROW_FUNC_ENTRANCE_LOG;
    uint8_t pixel_contents = 0;
    int row_data_high_wo_padding = 0, row_data_width_wo_padding = 0;
    
    row_data_high_wo_padding = row_data_high - padding;
    row_data_width_wo_padding = row_data_width - padding;

    if ((row_data_high_wo_padding<0) ||
        (row_data_width_wo_padding<0) ||
        (row_data_high_wo_padding >= in_highs) ||
        (row_data_width_wo_padding >= in_widths))
    {
        pixel_contents = 0;
    }
    else
    {
        pixel_contents = in_img[(input_channel_ind*in_highs+row_data_high_wo_padding)*in_widths + row_data_width_wo_padding];
    }
    IM2ROW_FUNC_EXIT_LOG;
    return pixel_contents;
};

static int inline row2im_get_im_ind(const int in_highs,
                                    const int in_widths,
                                    const int row_data_high,
                                    const int row_data_width,
                                    const int input_channel_ind,
                                    const int padding)
{
    IM2ROW_FUNC_ENTRANCE_LOG;
    int im_ind = 0;
    int row_data_high_wo_padding = 0, row_data_width_wo_padding = 0;

    row_data_high_wo_padding = row_data_high - padding;
    row_data_width_wo_padding = row_data_width - padding;

    if ((row_data_high_wo_padding<0) ||
        (row_data_width_wo_padding<0) || 
        (row_data_high_wo_padding>=in_highs) ||
        (row_data_width_wo_padding>=in_widths))
    {
        im_ind = IM2ROW_INVALID_IM_IND;
    }
    else 
    {
        im_ind = (input_channel_ind*in_highs + row_data_high_wo_padding)*in_widths + row_data_width_wo_padding;
    }
    IM2ROW_FUNC_EXIT_LOG;
    return im_ind;
};

void im2row_cpu_c(uint8_t const *in_img,
                  unsigned int const in_highs,
                  unsigned int const in_widths,
                  unsigned int const in_channels,
                  unsigned int const kernel_size,
                  unsigned int const stride,
                  unsigned int const padding,
                  uint8_t **pp_output_row);
void row2im_cpu_c(uint8_t const *in_row_data,
                  unsigned int const in_highs,
                  unsigned int const in_widths,
                  unsigned int const in_channels,
                  unsigned int const kernel_size,
                  unsigned int const stride,
                  unsigned int const padding,
                  uint8_t **pp_output_img);

static uint32_t inline im2row_get_pixel_uint32_t(uint32_t const *in_img,
                                                 const int in_highs,
                                                 const int in_widths,
                                                 const int row_data_high,
                                                 const int row_data_width,
                                                 const int input_channel_ind,
                                                 const int padding)
{
    IM2ROW_FUNC_ENTRANCE_LOG;
    uint32_t pixel_contents = 0;
    int row_data_high_wo_padding = 0, row_data_width_wo_padding = 0;
    
    row_data_high_wo_padding = row_data_high - padding;
    row_data_width_wo_padding = row_data_width - padding;

    if ((row_data_high_wo_padding<0) ||
        (row_data_width_wo_padding<0) ||
        (row_data_high_wo_padding >= in_highs) ||
        (row_data_width_wo_padding >= in_widths))
    {
        pixel_contents = 0;
    }
    else
    {
        pixel_contents = in_img[(input_channel_ind*in_highs+row_data_high_wo_padding)*in_widths + row_data_width_wo_padding];
    }
    IM2ROW_FUNC_EXIT_LOG;
    return pixel_contents;
};

void im2row_cpu_c_u32(uint32_t const *in_img,
                      unsigned int const in_highs,
                      unsigned int const in_widths,
                      unsigned int const in_channels,
                      unsigned int const kernel_size,
                      unsigned int const stride,
                      unsigned int const padding,
                      uint32_t **pp_output_row);
void row2im_cpu_c_u32(uint32_t const *in_row_data,
                      unsigned int const in_highs,
                      unsigned int const in_widths,
                      unsigned int const in_channels,
                      unsigned int const kernel_size,
                      unsigned int const stride,
                      unsigned int const padding,
                      uint32_t **pp_output_img);
#endif
