#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#include "android_arm_util.h"
#include "android_image_processing.h"
#include "android_histogram.h"

void histogram_gen(uint8_t const *raw_gray_img, uint32_t **pp_hist, unsigned int const pixel_num)
{
    FUNC_ENTRANCE_LOG;
    uint32_t *p_hist = NULL; 

    if ((!raw_gray_img) || (!pp_hist) || (pixel_num==0))
    {
        ree_log(LOG_ERROR, "%s occurs error due to invalid arguments", __func__);
        goto EXIT_HISTOGRAM_GEN;
    }

    p_hist = ree_malloc(sizeof(uint32_t)*CHANNEL_ITENSITY_LEVEL);
    ree_set(p_hist, 0, sizeof(uint32_t)*CHANNEL_ITENSITY_LEVEL);

    for (int i = 0; i<pixel_num; i++)
    {
        p_hist[raw_gray_img[i]]++;
    }
    *pp_hist = p_hist;
EXIT_HISTOGRAM_GEN:
    FUNC_EXIT_LOG;
}

void histogram_eq_map(uint32_t const *hist, uint8_t **pp_hist_map, unsigned int const pixel_num)
{
    FUNC_ENTRANCE_LOG;
    uint32_t count_accum = 0;
    uint8_t *p_hist_map = NULL;
    uint32_t *p_cdf = NULL;

    if ( (!hist) || (!pp_hist_map))
    {
        ree_log(LOG_ERROR, "%s occurs error with invalid parameters", __func__);
        goto EXIT_HISTOGRAM_EQ_MAP;
    }
    
    p_cdf = ree_malloc(sizeof(uint32_t)*CHANNEL_ITENSITY_LEVEL);
    ree_set(p_cdf, 0, sizeof(uint32_t)*CHANNEL_ITENSITY_LEVEL);

    if (!p_cdf)
    {
        ree_log(LOG_ERROR, "%s allocates p_cdb error", __func__);
        goto EXIT_HISTOGRAM_EQ_MAP;
    }

    p_hist_map = ree_malloc(sizeof(uint8_t)*CHANNEL_ITENSITY_LEVEL);
    ree_set(p_hist_map, 0, sizeof(uint8_t)*CHANNEL_ITENSITY_LEVEL);

    if (!p_hist_map)
    {
        ree_log(LOG_ERROR, "%s allocates p_hist_map error", __func__);
        goto EXIT_HISTOGRAM_EQ_MAP;
    }

    for (int i = 0; i<CHANNEL_ITENSITY_LEVEL; i++)
    {
        count_accum += hist[i];
        p_cdf[i] = count_accum;
    }

    for (int i = 0; i<CHANNEL_ITENSITY_LEVEL; i++)
    {
        p_hist_map[i] = (float)p_cdf[i]*255.0/(pixel_num);
    }
    *pp_hist_map = p_hist_map;
EXIT_HISTOGRAM_EQ_MAP:
    FUNC_EXIT_LOG;
}

void histogram_equalization_c(uint8_t const *raw_gray_img, 
                              uint8_t const *hist_map,
                              uint8_t **pp_hist_eq_res, 
                              unsigned int const pixel_num)
{
    FUNC_ENTRANCE_LOG;
    uint8_t *p_hist_eq_res = NULL;
    double start = 0.0f, end = 0.0f;
    
    if ((!raw_gray_img) || (!hist_map) || (!pp_hist_eq_res) || (pixel_num==0))
    {
        ree_log(LOG_ERROR, "%s occrus error with invalid parameter", __func__);
        goto EXIT_HISTOGRAM_EQUALIZATION_C;
    }

    p_hist_eq_res = ree_malloc(sizeof(uint8_t)*pixel_num);
    ree_set(p_hist_eq_res, 0, sizeof(uint8_t)*pixel_num);
    if (!p_hist_eq_res)
    {
        ree_log(LOG_ERROR, "%s allocates p_hist_eq_res failed", __func__);
        goto EXIT_HISTOGRAM_EQUALIZATION_C;
    }

    start = now_ns();
    for (int i = 0; i<pixel_num; i++)
    {
        p_hist_eq_res[i] = hist_map[raw_gray_img[i]];
    }
    end = now_ns();
    ree_log(LOG_DEBUG, "%s executes times %f", __func__, end - start);

    *pp_hist_eq_res = p_hist_eq_res;
EXIT_HISTOGRAM_EQUALIZATION_C:
    FUNC_EXIT_LOG;
}

void histogram_equalization_neon(uint8_t const *raw_gray_img,
                                 uint8_t const *hist_map,
                                 uint8_t **pp_hist_eq_res,
                                 unsigned int const pixel_num)
{
    FUNC_ENTRANCE_LOG;
    unsigned int num_iterator = 0;
    unsigned int residual = 0;
    double start = 0.0f, end = 0.0f;
    uint8_t *p_hist_eq_res = NULL;
    uint8x16_t vpixels;
    uint8x16_t vhist_res;
    uint8x16x4_t vhist_map[4];

    if ((!raw_gray_img) || (!hist_map) || (!pp_hist_eq_res) || (pixel_num==0))
    {
        ree_log(LOG_ERROR, "%s occurs error with invalid parameter", __func__);
        goto EXIT_HISTOGRAM_EQUALIZATION_NEON;
    }

    num_iterator = pixel_num/16;
    residual = pixel_num - num_iterator*16;
    ree_log(LOG_DEBUG, "%s num_iterator %d, residual %d", __func__, num_iterator, residual);
    
    p_hist_eq_res = ree_malloc(sizeof(uint8_t)*pixel_num);
    ree_set(p_hist_eq_res, 0, sizeof(uint8_t)*pixel_num);
    if (!p_hist_eq_res)
    {
        ree_log(LOG_ERROR, "%s allocates p_hist_eq_res failed", __func__);
        goto EXIT_HISTOGRAM_EQUALIZATION_NEON;
    }

    start = now_ns();
    for (int i = 0; i<HISTOGRAM_NEON_LOOKUPTBL_NUM; i++)
    {
        vhist_map[i].val[0] = vld1q_u8(hist_map + i*64);
        vhist_map[i].val[1] = vld1q_u8(hist_map + i*64 + 16);
        vhist_map[i].val[2] = vld1q_u8(hist_map + i*64 + 32);
        vhist_map[i].val[3] = vld1q_u8(hist_map + i*64 + 48);
    }


    for (int i = 0; i<num_iterator; i++)
    {
        vpixels = vld1q_u8(raw_gray_img + i*16);
        vhist_res = vqtbl4q_u8(vhist_map[0], vpixels);
        vhist_res = vqtbx4q_u8(vhist_res, vhist_map[1], vsubq_u8(vpixels, vdupq_n_u8(64)));
        vhist_res = vqtbx4q_u8(vhist_res, vhist_map[2], vsubq_u8(vpixels, vdupq_n_u8(128)));
        vhist_res = vqtbx4q_u8(vhist_res, vhist_map[3], vsubq_u8(vpixels, vdupq_n_u8(192)));
        vst1q_u8(p_hist_eq_res+i*16, vhist_res);
    }
    end = now_ns();
    ree_log(LOG_DEBUG, "%s executes times %f", __func__, end - start);

    *pp_hist_eq_res = p_hist_eq_res;
EXIT_HISTOGRAM_EQUALIZATION_NEON:
    FUNC_EXIT_LOG;
}

void histogram_match_map(uint8_t const *in_hist,
                         uint8_t const *ref_hist,
                         uint8_t **pp_hist_map)
{
    FUNC_ENTRANCE_LOG;
    uint8_t *p_hist_map = NULL;
    int s_i = 0, z_j = 0;
    unsigned int i = 0, j = 0;

    if ((!in_hist) || (!ref_hist) || (*pp_hist_map))
    {
        ree_log(LOG_ERROR, "%s with invalid parameters", __func__);
        goto EXIT_HISTOGRAM_MATCH_MAP;
    }

    p_hist_map = ree_malloc(sizeof(uint8_t)*CHANNEL_ITENSITY_LEVEL);
    ree_set(p_hist_map, 0, sizeof(uint8_t)*CHANNEL_ITENSITY_LEVEL);
    if (!p_hist_map)
    {
        ree_log(LOG_ERROR, "%s allocates p_hist_map error", __func__);
        goto EXIT_HISTOGRAM_MATCH_MAP;
    }
    ree_cpy(p_hist_map, in_hist, sizeof(uint8_t)*CHANNEL_ITENSITY_LEVEL);

    for (i = 0; i<CHANNEL_ITENSITY_LEVEL; i++)
    {
        s_i = in_hist[i];
        for (; j<CHANNEL_ITENSITY_LEVEL; j++)
        {
            z_j = ref_hist[j];
            if ((z_j - s_i)>0)
            {
                p_hist_map[i] = j-1;
                break;
            }
        }
    }
    *pp_hist_map = p_hist_map;
EXIT_HISTOGRAM_MATCH_MAP:
    FUNC_EXIT_LOG;
}

