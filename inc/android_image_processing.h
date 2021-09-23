#ifndef __ANDROID_IMAGE_PROCESSING_H__
#define __ANDROID_IMAGE_PROCESSING_H__
#define CHANNEL_ITENSITY_MIN         (0)
#define CHANNEL_ITENSITY_MAX         (5)
#define CHANNEL_ITENSITY_LEVEL       (256)

static int inline conv_2d_out_height(height, padding, kernel_size, stride)
{
    return (height + 2*padding - kernel_size) / stride + 1;
}

static int inline conv_2d_out_width(height, padding, kernel_size, stride)
{
    return (height + 2*padding - kernel_size) / stride + 1;
}

void c_hist_eq(uint8_t const *ori_img, uint8_t **pp_res_img, const int file_size);
void neon_hist_eq(uint8_t const *ori_img, uint8_t **pp_res_img, const int file_size);
void c_hist_match(uint8_t const *in_img,
                  uint8_t const *ref_img,
                  uint8_t **pp_in_res_img,
                  const unsigned int in_pixel_num,
                  const unsigned int ref_pixel_num);
void neon_hist_match(uint8_t const *in_img,
                     uint8_t const *ref_img,
                     uint8_t **pp_in_res_img,
                     const unsigned int in_pixel_num,
                     const unsigned int ref_pixel_num);
#endif
