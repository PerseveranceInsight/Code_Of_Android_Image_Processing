#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#include "android_arm_typedef.h"
#include "android_arm_util.h"
#include "android_image_processing_pack.h"

void pack_cpu_neon_uint32_t(uint32_t const *in_unpack_mat,
                            const int in_heights,
                            const int in_widths,
                            const int pack_heights,
                            const int pack_widths,
                            uint32_t **out_pack_mat)
{
    PACK_FUNC_ENTRANCE_LOG;
    int h_offset = 0, w_offset = 0;
    int st_offset = 0;
    int vec_head_ind[4] = {0};
    const int h_steps = in_heights/pack_heights + 1;
    const int w_steps = in_widths/pack_widths + 1;
    const int in_ele_num = in_heights*in_widths;
    const int out_ele_num = h_steps*w_steps*pack_heights*pack_widths;
    const int pack_sizes = pack_heights*pack_widths;
    uint32_t *p_unpack_mat = ree_malloc(out_ele_num*sizeof(uint32_t));
    uint32_t *p_pack_mat = ree_malloc(out_ele_num*sizeof(uint32_t));
    uint32x4x4_t pack_vec;
    ree_log(LOG_DEBUG, "h_steps %d", h_steps);
    ree_log(LOG_DEBUG, "w_steps %d", w_steps);
    ree_log(LOG_DEBUG, "in_ele_num %d", in_ele_num);
    ree_log(LOG_DEBUG, "out_ele_num %d", out_ele_num);
    ree_log(LOG_DEBUG, "pack_sizes %d", pack_sizes);

    if ((in_heights<=0) || (in_widths<=0) || (pack_heights<=0) || (pack_widths<=0))
    {
        ree_log(LOG_ERROR, "%s occurs error due to invalid parameters", __func__);
        goto EXIT_PACK_CPU_NEON_UINT32_T;
    }

    if ((h_steps<=0) || (w_steps<=0))
    {
        ree_log(LOG_ERROR, "%s occrus error due to invalid parameters", __func__);
        goto EXIT_PACK_CPU_NEON_UINT32_T;
    }

    if ((!p_unpack_mat) || (!p_pack_mat))
    {
        ree_log(LOG_ERROR, "%s allocates p_unpack_mat or p_pack_mat failed", __func__);
        goto EXIT_PACK_CPU_NEON_UINT32_T;
    }

    ree_set(p_unpack_mat, 0, sizeof(uint32_t)*out_ele_num);
    ree_set(p_pack_mat, 0, sizeof(uint32_t)*out_ele_num);
    ree_cpy(p_unpack_mat, in_unpack_mat, sizeof(uint32_t)*in_ele_num);

    for (int h_step = 0; h_step<h_steps; h_step++)
    {
        h_offset = h_step*pack_heights;
        // ree_log(LOG_DEBUG, "%s h_offset %d", __func__, h_offset);
        for (int w_step = 0; w_step<w_steps; w_step++)
        {
            w_offset = w_step*pack_widths;
            vec_head_ind[0] = h_offset*in_widths + w_offset;
            vec_head_ind[1] = vec_head_ind[0]+in_widths;
            vec_head_ind[2] = vec_head_ind[1]+in_widths;
            vec_head_ind[3] = vec_head_ind[2]+in_widths;
            // ree_log(LOG_DEBUG, "%s w_offset %d", __func__, w_offset);
            // ree_log(LOG_DEBUG, "%s vec_head_ind %d %d %d %d", __func__, vec_head_ind[0],
            //                                                             vec_head_ind[1],
            //                                                             vec_head_ind[2],
            //                                                             vec_head_ind[3]);
            pack_vec.val[0] = vld1q_u32(p_unpack_mat+vec_head_ind[0]);
            pack_vec.val[1] = vld1q_u32(p_unpack_mat+vec_head_ind[1]);
            pack_vec.val[2] = vld1q_u32(p_unpack_mat+vec_head_ind[2]);
            pack_vec.val[3] = vld1q_u32(p_unpack_mat+vec_head_ind[3]);
            vst4q_u32(p_pack_mat+st_offset, pack_vec);
            st_offset += pack_sizes;
        }
    }
    *out_pack_mat = p_pack_mat;

EXIT_PACK_CPU_NEON_UINT32_T:
    ree_free(p_unpack_mat);
    PACK_FUNC_EXIT_LOG;
}
