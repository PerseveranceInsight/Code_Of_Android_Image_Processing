#ifndef __ANDROID_IMAGE_PROCESSING_IM2COL_H__
#define __ANDROID_IMAGE_PROCESSING_IM2COL_H__
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
#endif
