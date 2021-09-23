#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h> 

#include "android_image_processing.h"
#include "android_histogram.h"
#include "android_arm_util.h"

void c_hist_eq(uint8_t const *ori_img, uint8_t **pp_res_img, const int file_size)
{
    FUNC_ENTRANCE_LOG;
    int pixel_num = file_size;
    uint8_t *hist_map = NULL;
    uint8_t *hist_eq_res = NULL;
    uint32_t *hist = NULL;
    
    if ((!ori_img) || (*pp_res_img) || (file_size<=0))
    {
        ree_log(LOG_ERROR, "%s with invalid parameters", __func__);
        goto EXIT_C_HIST_EQ;
    }
    ree_log(LOG_DEBUG, "%s pixel_num %d", __func__, pixel_num);
    
    histogram_gen(ori_img, &hist, pixel_num);

#ifdef ANDROID_IPP_DEBUG_EXP_LOG
    if (hist) 
    {
        for (int i = 0; i < CHANNEL_ITENSITY_LEVEL; i++)
        {
            ree_log(LOG_DEBUG, "%s hist[%d]=%d", __func__, i,  hist[i]);
        }
    }
#endif
    histogram_eq_map(hist, &hist_map, pixel_num);

#ifdef ANDROID_IPP_DEBUG_EXP_LOG
    if (hist_map)
    {
        for (int i = 0; i < CHANNEL_ITENSITY_LEVEL; i++)
        {
            ree_log(LOG_DEBUG, "%s hist_map[%d]=%d", __func__, i,  hist_map[i]);
        }
    }
#endif
    histogram_equalization_c(ori_img, hist_map, &hist_eq_res, pixel_num);
    *pp_res_img = hist_eq_res;

EXIT_C_HIST_EQ:
    ree_free(hist);
    ree_free(hist_map);
    FUNC_EXIT_LOG;
}

void neon_hist_eq(uint8_t const *ori_img, uint8_t **pp_res_img, const int file_size)
{
    FUNC_ENTRANCE_LOG;
    int pixel_num = file_size;
    uint8_t *hist_map = NULL;
    uint8_t *hist_eq_res = NULL;
    uint32_t *hist = NULL;
    
    if ((!ori_img) || (*pp_res_img) || (file_size<=0))
    {
        ree_log(LOG_ERROR, "%s with invalid parameters", __func__);
        goto EXIT_NEON_HIST_EQ;
    }
    ree_log(LOG_DEBUG, "%s pixel_num %d", __func__, pixel_num);
    
    histogram_gen(ori_img, &hist, pixel_num);

#ifdef ANDROID_IPP_DEBUG_EXP_LOG
    if (hist) 
    {
        for (int i = 0; i < CHANNEL_ITENSITY_LEVEL; i++)
        {
            ree_log(LOG_DEBUG, "%s hist[%d]=%d", __func__, i,  hist[i]);
        }
    }
#endif
    histogram_eq_map(hist, &hist_map, pixel_num);

#ifdef ANDROID_IPP_DEBUG_EXP_LOG
    if (hist_map)
    {
        for (int i = 0; i < CHANNEL_ITENSITY_LEVEL; i++)
        {
            ree_log(LOG_DEBUG, "%s hist_map[%d]=%d", __func__, i,  hist_map[i]);
        }
    }
#endif
    histogram_equalization_neon(ori_img, hist_map, &hist_eq_res, pixel_num);
    *pp_res_img = hist_eq_res;

EXIT_NEON_HIST_EQ:
    ree_free(hist);
    ree_free(hist_map);
    FUNC_EXIT_LOG;
}

void c_hist_match(uint8_t const *in_img,
                  uint8_t const *ref_img,
                  uint8_t **pp_in_res_img,
                  const unsigned int in_pixel_num,
                  const unsigned int ref_pixel_num)
{
    FUNC_ENTRANCE_LOG;
    uint8_t *in_hist_eq_map = NULL;
    uint8_t *ref_hist_eq_map = NULL;   
    uint8_t *in_hist_match_map = NULL;
    uint8_t *in_hist_match_res = NULL;
    uint32_t *in_img_hist = NULL;
    uint32_t *ref_img_hist = NULL;
    
    if ((!in_img) || (in_pixel_num==0))
    {
        ree_log(LOG_ERROR, "%s with invalid parameters", __func__);
        goto EXIT_C_HIST_MATCH;
    }

    ree_log(LOG_DEBUG, "%s in_file_size, pixel_num %d", __func__, in_pixel_num);
    ree_log(LOG_DEBUG, "%s ref_file_size, pixel_num %d", __func__, ref_pixel_num);
    histogram_gen(in_img, &in_img_hist, in_pixel_num);
    histogram_gen(ref_img, &ref_img_hist, ref_pixel_num);
    histogram_eq_map(in_img_hist, &in_hist_eq_map, in_pixel_num);
    histogram_eq_map(ref_img_hist, &ref_hist_eq_map, ref_pixel_num);
    histogram_match_map(in_hist_eq_map,
                        ref_hist_eq_map,
                        &in_hist_match_map);
    histogram_equalization_c(in_img, in_hist_match_map, &in_hist_match_res, in_pixel_num);
    *pp_in_res_img = in_hist_match_res;


 EXIT_C_HIST_MATCH:
    ree_free(in_img_hist);
    ree_free(ref_img_hist);
    ree_free(in_hist_eq_map);
    ree_free(ref_hist_eq_map);
    ree_free(in_hist_match_map);
    FUNC_EXIT_LOG;
}

void neon_hist_match(uint8_t const *in_img,
                     uint8_t const *ref_img,
                     uint8_t **pp_in_res_img,
                     const unsigned int in_pixel_num,
                     const unsigned int ref_pixel_num)
{
    FUNC_ENTRANCE_LOG;
    uint8_t *in_hist_eq_map = NULL;
    uint8_t *ref_hist_eq_map = NULL;   
    uint8_t *in_hist_match_map = NULL;
    uint8_t *in_hist_match_res = NULL;
    uint32_t *in_img_hist = NULL;
    uint32_t *ref_img_hist = NULL;

    if ((!in_img) || (in_pixel_num==0))
    {
        ree_log(LOG_ERROR, "%s with invalid parameters", __func__);
        goto EXIT_NEON_HIST_MATCH;
    }

    ree_log(LOG_DEBUG, "%s in_file_size, pixel_num %d", __func__, in_pixel_num);
    ree_log(LOG_DEBUG, "%s ref_file_size, pixel_num %d", __func__, ref_pixel_num);
    histogram_gen(in_img, &in_img_hist, in_pixel_num);
    histogram_gen(ref_img, &ref_img_hist, ref_pixel_num);
    histogram_eq_map(in_img_hist, &in_hist_eq_map, in_pixel_num);
    histogram_eq_map(ref_img_hist, &ref_hist_eq_map, ref_pixel_num);
    histogram_match_map(in_hist_eq_map,
                        ref_hist_eq_map,
                        &in_hist_match_map);
    histogram_equalization_neon(in_img, in_hist_match_map, &in_hist_match_res, in_pixel_num);
    *pp_in_res_img = in_hist_match_res;

EXIT_NEON_HIST_MATCH:
    ree_free(in_img_hist);
    ree_free(ref_img_hist);
    ree_free(in_hist_eq_map);
    ree_free(ref_hist_eq_map);
    ree_free(in_hist_match_map);
    FUNC_EXIT_LOG;
}
