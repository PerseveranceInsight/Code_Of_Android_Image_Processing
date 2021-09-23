#ifndef __ANDROID_IMAGE_PROCESSING_HISTOGRAM_H__
#define __ANDROID_IMAGE_PROCESSING_HISTOGRAM_H__
#define HISTOGRAM_NEON_LOOKUPTBL_NUM        (4)
void histogram_gen(uint8_t const *raw_gray_img, uint32_t **pp_hist, unsigned int const pixel_num);
void histogram_eq_map(uint32_t const *hist, uint8_t **pp_hist_map, unsigned int const pixel_num);
void histogram_match_map(uint8_t const *in_hist,
                         uint8_t const *ref_hist,
                         uint8_t **pp_hist_map);
void histogram_equalization_c(uint8_t const *raw_gray_img, 
                              uint8_t const *hist_map,
                              uint8_t **pp_hist_eq_res, 
                              unsigned int const pixel_num);
void histogram_equalization_neon(uint8_t const *raw_gray_img,
                                 uint8_t const *hist_map,
                                 uint8_t **pp_hist_eq_res,
                                 unsigned int const in_pixel_num);
#endif
