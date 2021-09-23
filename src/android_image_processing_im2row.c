#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "android_arm_util.h"
#include "android_image_processing_im2row.h"

void im2row_cpu_c(uint8_t const *in_img,
                  unsigned int const in_highs,
                  unsigned int const in_widths,
                  unsigned int const in_channels,
                  unsigned int const kernel_size,
                  unsigned int const stride,
                  unsigned int const padding,
                  uint8_t **pp_output_row)
{
    IM2ROW_FUNC_ENTRANCE_LOG;
    uint8_t *p_row_data = NULL;
    int output_highs = 0, output_widths = 0, output_channels = 0, output_nums = 0;
    int high_offset = 0, width_offset = 0, input_channel_ind = 0;
    int row_data_ind = 0;

    output_highs = (in_highs + 2*padding - kernel_size) / stride + 1;
    output_widths = (in_widths + 2*padding - kernel_size) / stride + 1;
    output_channels = kernel_size*kernel_size*in_channels;
    output_nums = output_highs * output_widths;

#ifdef EN_IM2ROW_DEBUG
    ree_log(LOG_DEBUG, "%s output_highs %d", __func__, output_highs);
    ree_log(LOG_DEBUG, "%s output_widths %d", __func__, output_widths);
    ree_log(LOG_DEBUG, "%s output_num %d", __func__, output_nums);
    ree_log(LOG_DEBUG, "%s output_channels %d", __func__, output_channels);
#endif

    p_row_data = ree_malloc(sizeof(uint8_t)*output_nums*output_channels);
    if (!p_row_data)
    {
        ree_log(LOG_ERROR, "%s allocs p_row_data failed", __func__);
        goto EXIT_IM2ROW_CPU_C;
    }
    ree_set(p_row_data, 0, sizeof(uint8_t)*output_nums*output_channels);

    for (int output_channels_ind = 0; output_channels_ind<output_channels; output_channels_ind++)
    {
        for(int output_high_ind = 0; output_high_ind<output_highs; output_high_ind++)
        {
            for (int output_width_ind = 0; output_width_ind<output_widths; output_width_ind++)
            {
                width_offset = output_channels_ind%kernel_size;
                high_offset = (output_channels_ind/kernel_size)%kernel_size;
                input_channel_ind = output_channels_ind / (kernel_size*kernel_size);
                row_data_ind = output_channels_ind + (output_width_ind + output_high_ind*output_widths)*output_channels;
#ifdef EN_IM2ROW_DEBUG
                ree_log(LOG_DEBUG, "%s row_data_ind %d", __func__, row_data_ind);
#endif
                p_row_data[row_data_ind] = im2row_get_pixel_uint8_t(in_img,
                                                                    in_highs, in_widths,
                                                                    high_offset + output_high_ind*stride,
                                                                    width_offset + output_width_ind*stride,
                                                                    input_channel_ind,
                                                                    padding);
            }
        }
    }

#ifdef EN_IM2ROW_DEBUG
    for (int output_high_ind = 0; output_high_ind<output_highs; output_high_ind++)
    {
        for (int output_width_ind = 0; output_width_ind<output_widths; output_width_ind++)
        {
            for (int output_channel_ind = 0; output_channel_ind<output_channels; output_channel_ind++)
            {
                row_data_ind = output_channel_ind + (output_high_ind*output_widths + output_width_ind)*output_channels;
                ree_printf(LOG_DEBUG, "%d ", p_row_data[row_data_ind]);
            }
            ree_printf(LOG_DEBUG, "\n");
        }
    }
#endif
    *pp_output_row = p_row_data;
EXIT_IM2ROW_CPU_C:
    IM2ROW_FUNC_EXIT_LOG;
}

void row2im_cpu_c(uint8_t const *in_row_data,
                  unsigned int const in_highs,
                  unsigned int const in_widths,
                  unsigned int const in_channels,
                  unsigned int const kernel_size,
                  unsigned int const stride,
                  unsigned int const padding,
                  uint8_t **pp_output_img)
{
    IM2ROW_FUNC_ENTRANCE_LOG;
    uint8_t *p_output_img = NULL;
    unsigned int output_row_highs = 0, output_row_widths = 0, output_row_channels = 0;
    unsigned int high_offset = 0, width_offset = 0, input_channel_ind = 0;
    unsigned int row_data_ind = 0;
    int output_img_ind = 0;

    if (!in_row_data) {
        ree_log(LOG_ERROR, "%s occurs error due to invalid parameter", __func__);
        goto EXIT_ROW2IM_CPU_C;
    }
    
    output_row_highs =  (in_highs + 2*padding - kernel_size)/stride + 1;
    output_row_widths = (in_widths + 2*padding - kernel_size)/stride + 1;
    output_row_channels = (kernel_size*kernel_size)*in_channels;

#ifdef EN_IM2ROW_DEBUG
    ree_log(LOG_DEBUG, "%s output_row_highs %d", __func__, output_row_highs);
    ree_log(LOG_DEBUG, "%s output_row_widths %d", __func__, output_row_widths);
    ree_log(LOG_DEBUG, "%s output_row_channels %d", __func__, output_row_channels);
#endif

    p_output_img = ree_malloc(sizeof(uint8_t)*in_highs*in_widths*in_channels);
    if (!p_output_img)
    {
        ree_log(LOG_DEBUG, "%s allocates p_output_img failed", __func__);
        goto EXIT_ROW2IM_CPU_C;
    }
    ree_set(p_output_img, 0, sizeof(uint8_t)*in_highs*in_widths*in_channels);

    for (int output_channel_ind = 0; output_channel_ind < output_row_channels; output_channel_ind++)
    {
        width_offset = (output_channel_ind) % kernel_size;
        high_offset = (output_channel_ind / kernel_size) % kernel_size;
        input_channel_ind = output_channel_ind / (kernel_size*kernel_size);
        for (int output_row_high_ind = 0; output_row_high_ind < output_row_highs; output_row_high_ind++)
        {
            for (int output_row_width_ind  = 0; output_row_width_ind < output_row_widths; output_row_width_ind++)
            {
               row_data_ind = output_channel_ind + (output_row_width_ind + output_row_high_ind * output_row_widths)*output_row_highs;
#ifdef EN_IM2ROW_DEBUG
               ree_log(LOG_DEBUG, "%s row_data_ind %d", __func__, row_data_ind);
#endif
               output_img_ind = row2im_get_im_ind(in_highs, in_widths,
                                                  high_offset + output_row_high_ind*stride,
                                                  width_offset + output_row_width_ind*stride,
                                                  input_channel_ind,
                                                  padding);
               if (output_img_ind!=IM2ROW_INVALID_IM_IND)
               {
#ifdef EN_IM2ROW_DEBUG
                    ree_log(LOG_DEBUG, "%s output_img_ind %d %d", __func__, row_data_ind, output_img_ind);
#endif
                    p_output_img[output_img_ind] = in_row_data[row_data_ind]; 
               }
            }
        }
    }

#ifdef EN_IM2ROW_DEBUG
    for (int high = 0; high<in_highs; high++)
    {
        for (int width = 0; width<in_widths; width++)
        {
            ree_printf(LOG_DEBUG, "%d ", p_output_img[width+high*in_widths]);
        }
        ree_printf(LOG_DEBUG, "\n");
    }
#endif

    *pp_output_img = p_output_img;
EXIT_ROW2IM_CPU_C:
    IM2ROW_FUNC_EXIT_LOG;
}

void im2row_cpu_c_u32(uint32_t const *in_img,
                      unsigned int const in_highs,
                      unsigned int const in_widths,
                      unsigned int const in_channels,
                      unsigned int const kernel_size,
                      unsigned int const stride,
                      unsigned int const padding,
                      uint32_t **pp_output_row)
{
    IM2ROW_FUNC_ENTRANCE_LOG;
    uint32_t *p_row_data = NULL;
    int output_highs = 0, output_widths = 0, output_channels = 0, output_nums = 0;
    int high_offset = 0, width_offset = 0, input_channel_ind = 0;
    int row_data_ind = 0;

    output_highs = (in_highs + 2*padding - kernel_size) / stride + 1;
    output_widths = (in_widths + 2*padding - kernel_size) / stride + 1;
    output_channels = kernel_size*kernel_size*in_channels;
    output_nums = output_highs * output_widths;

#ifdef EN_IM2ROW_DEBUG
    ree_log(LOG_DEBUG, "%s output_highs %d", __func__, output_highs);
    ree_log(LOG_DEBUG, "%s output_widths %d", __func__, output_widths);
    ree_log(LOG_DEBUG, "%s output_num %d", __func__, output_nums);
    ree_log(LOG_DEBUG, "%s output_channels %d", __func__, output_channels);
#endif

    p_row_data = ree_malloc(sizeof(uint32_t)*output_nums*output_channels);
    if (!p_row_data)
    {
        ree_log(LOG_ERROR, "%s allocs p_row_data failed", __func__);
        goto EXIT_IM2ROW_CPU_C_UINT32;
    }
    ree_set(p_row_data, 0, sizeof(uint32_t)*output_nums*output_channels);

    for (int output_channels_ind = 0; output_channels_ind<output_channels; output_channels_ind++)
    {
        for(int output_high_ind = 0; output_high_ind<output_highs; output_high_ind++)
        {
            for (int output_width_ind = 0; output_width_ind<output_widths; output_width_ind++)
            {
                width_offset = output_channels_ind%kernel_size;
                high_offset = (output_channels_ind/kernel_size)%kernel_size;
                input_channel_ind = output_channels_ind / (kernel_size*kernel_size);
                row_data_ind = output_channels_ind + (output_width_ind + output_high_ind*output_widths)*output_channels;
#ifdef EN_IM2ROW_DEBUG
                ree_log(LOG_DEBUG, "%s row_data_ind %d", __func__, row_data_ind);
#endif
                p_row_data[row_data_ind] = im2row_get_pixel_uint32_t(in_img,
                                                                     in_highs, in_widths,
                                                                     high_offset + output_high_ind*stride,
                                                                     width_offset + output_width_ind*stride,
                                                                     input_channel_ind,
                                                                     padding);
            }
        }
    }

#ifdef EN_IM2ROW_DEBUG
    for (int output_high_ind = 0; output_high_ind<output_highs; output_high_ind++)
    {
        for (int output_width_ind = 0; output_width_ind<output_widths; output_width_ind++)
        {
            for (int output_channel_ind = 0; output_channel_ind<output_channels; output_channel_ind++)
            {
                row_data_ind = output_channel_ind + (output_high_ind*output_widths + output_width_ind)*output_channels;
                ree_printf(LOG_DEBUG, "%d ", p_row_data[row_data_ind]);
            }
            ree_printf(LOG_DEBUG, "\n");
        }
    }
#endif
    *pp_output_row = p_row_data;
EXIT_IM2ROW_CPU_C_UINT32:
    IM2ROW_FUNC_EXIT_LOG;
}


