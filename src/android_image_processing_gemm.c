#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#include "android_arm_util.h"
#include "android_arm_typedef.h"
#include "android_image_processing_gemm.h"

static void gemm_cpu_c_nn_uint32(gemm_param_submetadata_t *p_submetadata)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    int m_ind = 0, n_ind = 0, k_ind = 0;
    int m_dims = p_submetadata->m_dims;
    int n_dims = p_submetadata->n_dims;
    int k_dims = p_submetadata->k_dims;
    int alpha = p_submetadata->alpha;
    int ele_a = 0;
    uint32_t *p_mat_a = (uint32_t*)p_submetadata->mat_a;
    uint32_t *p_mat_b = (uint32_t*)p_submetadata->mat_b;
    uint32_t *p_mat_c = (uint32_t*)p_submetadata->mat_c;

    for (m_ind = 0; m_ind<m_dims;m_ind++)
    {
        for (k_ind = 0; k_ind<k_dims; k_ind++)
        {
            ele_a = alpha * p_mat_a[m_ind*k_dims + k_ind];
            for (n_ind = 0; n_ind<n_dims; n_ind++)
            {
                p_mat_c[m_ind*n_dims + n_ind] += ele_a * p_mat_b[k_ind*n_dims+n_ind];
            }
        }
    }
    ANDROID_GEMM_FUNC_EXIT_LOG;
}

static void gemm_cpu_c_nt_uint32(gemm_param_submetadata_t *p_submetadata)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    int m_ind = 0, n_ind = 0, k_ind = 0;
    int m_dims = p_submetadata->m_dims;
    int n_dims = p_submetadata->n_dims;
    int k_dims = p_submetadata->k_dims;
    int alpha = p_submetadata->alpha;
    int micro_kernel_sum = 0;
    uint32_t *p_mat_a = (uint32_t*)p_submetadata->mat_a;
    uint32_t *p_mat_b = (uint32_t*)p_submetadata->mat_b;
    uint32_t *p_mat_c = (uint32_t*)p_submetadata->mat_c;

    double start = 0.0f, end = 0.0f;

    start = now_ns();
    for (m_ind = 0; m_ind<m_dims; m_ind++)
    {
        for (n_ind = 0; n_ind<n_dims; n_ind++)
        {
            micro_kernel_sum = 0;
            for (k_ind = 0; k_ind<k_dims; k_ind++)
            {
                micro_kernel_sum += alpha * p_mat_a[m_ind*k_dims+k_ind] * p_mat_b[n_ind*k_dims+k_ind];
            }
            p_mat_c[m_ind*n_dims+n_ind] += micro_kernel_sum;
        }
    }
    end = now_ns();
    ree_log(LOG_DEBUG, "%s exe times %f", __func__, end-start);
    ANDROID_GEMM_FUNC_EXIT_LOG;
}

static void gemm_cpu_c_tn_uint32(gemm_param_submetadata_t *p_submetadata)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    int m_ind = 0, n_ind = 0, k_ind = 0;
    int m_dims = p_submetadata->m_dims;
    int n_dims = p_submetadata->n_dims;
    int k_dims = p_submetadata->k_dims;
    int alpha = p_submetadata->alpha;
    int ele_a = 0;
    uint32_t *p_mat_a = (uint32_t*)p_submetadata->mat_a;
    uint32_t *p_mat_b = (uint32_t*)p_submetadata->mat_b;
    uint32_t *p_mat_c = (uint32_t*)p_submetadata->mat_c;

    for (m_ind = 0; m_ind<m_dims; m_ind++)
    {
        for (k_ind = 0; k_ind<k_dims; k_ind++)
        {
            ele_a = alpha * p_mat_a[k_ind*m_dims+m_ind];
            for (n_ind = 0; n_ind<n_dims; n_ind++)
            {
                p_mat_c[m_ind*n_dims+n_ind] += ele_a*p_mat_b[k_ind*n_dims+n_ind];
            }
        }
    }

    ANDROID_GEMM_FUNC_EXIT_LOG;
}

static void gemm_cpu_c_tt_uint32(gemm_param_submetadata_t *p_submetadata)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    int m_ind = 0, n_ind = 0, k_ind = 0;
    int m_dims = p_submetadata->m_dims;
    int n_dims = p_submetadata->n_dims;
    int k_dims = p_submetadata->k_dims;
    int alpha = p_submetadata->alpha;
    int micro_kernel_sum = 0;
    uint32_t *p_mat_a = (uint32_t*)p_submetadata->mat_a;
    uint32_t *p_mat_b = (uint32_t*)p_submetadata->mat_b;
    uint32_t *p_mat_c = (uint32_t*)p_submetadata->mat_c;

    for (m_ind = 0; m_ind<m_dims; m_ind++)
    {
        for (n_ind = 0; n_ind<n_dims; n_ind++)
        {
            micro_kernel_sum = 0;
            for (k_ind = 0; k_ind<k_dims; k_ind++)
            {
                micro_kernel_sum += alpha * p_mat_a[k_ind*m_dims+m_ind]*p_mat_b[n_ind*k_dims+k_ind];
            }
            p_mat_c[m_ind*n_dims+n_ind] += micro_kernel_sum;
        }
    }

    ANDROID_GEMM_FUNC_EXIT_LOG;
}

void gemm_cpu_c_uint32(gemm_param_metadata_t *p_metadata)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    BOOL is_a_tran = p_metadata->is_a_tran;
    BOOL is_b_tran = p_metadata->is_b_tran;
    int m_ind = 0, n_ind = 0, k_ind = 0;
    int m_dims = p_metadata->m_dims;
    int n_dims = p_metadata->n_dims;
    int k_dims = p_metadata->k_dims;
    int alpha = p_metadata->alpha;
    int beta = p_metadata->beta;
    uint32_t *p_mat_a = NULL;
    uint32_t *p_mat_b = NULL;
    uint32_t *p_mat_c = NULL;
    gemm_param_submetadata_t sub_param = {0};

    ree_log(LOG_DEBUG, "%s m_dims %d", __func__, m_dims);
    ree_log(LOG_DEBUG, "%s k_dims %d", __func__, k_dims);
    ree_log(LOG_DEBUG, "%s n_dims %d", __func__, n_dims);
    
    if (!p_metadata)
    {
        ree_log(LOG_ERROR, "%s occurs error with invalid parameters", __func__);
        goto EXIT_GEMM_CPU_C;
    }

    if ((!p_metadata->mat_a)||(!p_metadata->mat_b)||(!p_metadata->mat_c))
    {
        ree_log(LOG_ERROR, "%s occurs error with invalid parameters NULL MATRIX", __func__);
        goto EXIT_GEMM_CPU_C;
    }

    if ((m_dims<=0)||(n_dims<=0)||(k_dims<=0))
    {
        ree_log(LOG_ERROR, "%s occurs error with invalid parameters 0 dimension", __func__);
        goto EXIT_GEMM_CPU_C;
    }

    p_mat_a = (uint32_t*)p_metadata->mat_a;
    p_mat_b = (uint32_t*)p_metadata->mat_b;
    p_mat_c = (uint32_t*)p_metadata->mat_c;
   
    for (m_ind = 0;m_ind<m_dims;m_ind++)
    {
        for (n_ind = 0;n_ind<n_dims;n_ind++)
        {
            p_mat_c[m_ind*n_dims+n_ind] += beta*p_mat_c[m_ind*n_dims+n_ind];
        }
    }
    sub_param.m_dims = m_dims;
    sub_param.n_dims = n_dims;
    sub_param.k_dims = k_dims;
    sub_param.alpha = alpha;
    sub_param.mat_a = (uint8_t*)p_mat_a;
    sub_param.mat_b = (uint8_t*)p_mat_b;
    sub_param.mat_c = (uint8_t*)p_mat_c;

    if ((!is_a_tran)&&(!is_b_tran))
    {
        ree_log(LOG_INFO, "%s gemm_cpu_c N-N", __func__);
        gemm_cpu_c_nn_uint32(&sub_param);
    }
    else if ((!is_a_tran)&&(is_b_tran))
    {
        ree_log(LOG_INFO, "%s gemm_cpu_c N-T", __func__);
        gemm_cpu_c_nt_uint32(&sub_param);
    }
    else if ((is_a_tran)&&(!is_b_tran))
    {
        ree_log(LOG_INFO, "%s gemm_cpu_c T-N", __func__);
        gemm_cpu_c_tn_uint32(&sub_param);
    }
    else if ((is_a_tran)&&(is_b_tran))
    {
        ree_log(LOG_INFO, "%s gemm_cpu_c T-T", __func__);
        gemm_cpu_c_tt_uint32(&sub_param);
    }

EXIT_GEMM_CPU_C:
    ANDROID_GEMM_FUNC_EXIT_LOG;
}

static void gemm_cpu_accu_neon_uint32(gemm_pack_param_t *p_pack_params)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    int m_dims = p_pack_params->m_dims;
    int n_dims = p_pack_params->n_dims;
    int pack_heights = p_pack_params->pack_heights;
    int pack_widths = p_pack_params->pack_widths;
    int m_steps = m_dims/pack_heights+1;
    int n_steps = n_dims/pack_widths+1;
    int m_offset = 0, n_offset = 0;
    int out_aux_ele_num = m_steps*n_steps*pack_widths*pack_heights;
    int remain_lanes = pack_widths - (n_steps*pack_widths - n_dims);
    int pack_ori_ind_0 = 0, pack_ori_ind_1 = 0, pack_ori_ind_2 = 0, pack_ori_ind_3 = 0;
    uint32_t *p_mat_c = (uint32_t*)p_pack_params->mat_c;
    uint32_t *p_mat_aux_c = NULL;
    uint32_t *p_vec_remain_beta = NULL;
    uint32x4x4_t pack_vec_c;
    uint32x4x4_t pack_vec_beta;
    uint32x4x4_t pack_vec_remain_beta;
    ree_log(LOG_DEBUG, "%s m_dims %d", __func__, m_dims);
    ree_log(LOG_DEBUG, "%s n_dims %d", __func__, n_dims);
    ree_log(LOG_DEBUG, "%s m_steps %d", __func__, m_steps);
    ree_log(LOG_DEBUG, "%s n_steps %d", __func__, n_steps);
    ree_log(LOG_DEBUG, "%s pack_heights %d", __func__, pack_heights);
    ree_log(LOG_DEBUG, "%s pack_widths %d", __func__, pack_widths);
    ree_log(LOG_DEBUG, "%s beta %d", __func__, p_pack_params->beta);
    ree_log(LOG_DEBUG, "%s out_aux_ele_num %d", __func__, out_aux_ele_num);
    ree_log(LOG_DEBUG, "%s remain_lanes %d", __func__, remain_lanes);

    
    p_pack_params->mat_c_aux = (uint8_t*)ree_malloc(sizeof(uint32_t)*out_aux_ele_num);
    p_mat_aux_c = (uint32_t*)p_pack_params->mat_c_aux;

    if (!p_mat_aux_c)
    {
        ree_log(LOG_ERROR, "%s allocates p_pack_params->mat_c_aux failed", __func__);
        goto EXIT_GEMM_CPU_ACCU_NEON_UINT32;
    }

    p_vec_remain_beta = (uint32_t*)ree_malloc(sizeof(uint32_t)*pack_widths);

    if (!p_vec_remain_beta)
    {
        ree_log(LOG_ERROR, "%s allocates p_vec_remain_beta failed", __func__);
        goto EXIT_GEMM_CPU_ACCU_NEON_UINT32;
    }
    ree_log(LOG_DEBUG, "246");
    ree_cpy(p_mat_aux_c,
            p_mat_c,
            sizeof(uint32_t)*m_dims*n_dims);

    if (p_pack_params->beta==0)
    {
        ree_log(LOG_DEBUG, "%s beta==0 directly return", __func__);
        goto EXIT_GEMM_CPU_ACCU_NEON_UINT32;
    }

    ree_set(p_vec_remain_beta,
            0,
            pack_widths*sizeof(uint32_t));
    for(int i = 0; i<remain_lanes; i++)
        p_vec_remain_beta[i] = p_pack_params->beta+1;

    pack_vec_beta.val[0] = vdupq_n_u32(p_pack_params->beta+1);
    pack_vec_beta.val[1] = vdupq_n_u32(p_pack_params->beta+1);
    pack_vec_beta.val[2] = vdupq_n_u32(p_pack_params->beta+1);
    pack_vec_beta.val[3] = vdupq_n_u32(p_pack_params->beta+1);
    pack_vec_remain_beta.val[0] = vld1q_u32(p_vec_remain_beta);
    pack_vec_remain_beta.val[1] = vld1q_u32(p_vec_remain_beta);
    pack_vec_remain_beta.val[2] = vld1q_u32(p_vec_remain_beta);
    pack_vec_remain_beta.val[3] = vld1q_u32(p_vec_remain_beta);


    for (int m_step = 0; m_step<m_steps; m_step++)
    {
        m_offset = m_step * pack_heights;
        for (int n_step = 0; n_step<(n_steps-1); n_step++)
        {
            n_offset = n_step * pack_widths;
            pack_ori_ind_0 = m_offset*n_dims+n_offset;
            pack_ori_ind_1 = (m_offset+1)*n_dims+n_offset;
            pack_ori_ind_2 = (m_offset+1)*n_dims+n_offset;
            pack_ori_ind_3 = (m_offset+2)*n_dims+n_offset;
            pack_vec_c.val[0] = vld1q_u32((uint32_t*)(p_mat_aux_c+pack_ori_ind_0));
            pack_vec_c.val[1] = vld1q_u32((uint32_t*)(p_mat_aux_c+pack_ori_ind_1));
            pack_vec_c.val[2] = vld1q_u32((uint32_t*)(p_mat_aux_c+pack_ori_ind_2));
            pack_vec_c.val[3] = vld1q_u32((uint32_t*)(p_mat_aux_c+pack_ori_ind_3));

        }
        n_offset = (n_steps-1) * pack_widths;
        pack_ori_ind_0 = m_offset*n_dims+n_offset;
        pack_ori_ind_1 = (m_offset+1)*n_dims+n_offset;
        pack_ori_ind_2 = (m_offset+1)*n_dims+n_offset;
        pack_ori_ind_3 = (m_offset+2)*n_dims+n_offset;
        pack_vec_c.val[0] = vld1q_u32((uint32_t*)(p_mat_aux_c+pack_ori_ind_0));
        pack_vec_c.val[1] = vld1q_u32((uint32_t*)(p_mat_aux_c+pack_ori_ind_1));
        pack_vec_c.val[2] = vld1q_u32((uint32_t*)(p_mat_aux_c+pack_ori_ind_2));
        pack_vec_c.val[3] = vld1q_u32((uint32_t*)(p_mat_aux_c+pack_ori_ind_3));
        pack_vec_c.val[0] = vmulq_u32(pack_vec_c.val[0], pack_vec_remain_beta.val[0]);
        pack_vec_c.val[1] = vmulq_u32(pack_vec_c.val[1], pack_vec_remain_beta.val[1]);
        pack_vec_c.val[2] = vmulq_u32(pack_vec_c.val[2], pack_vec_remain_beta.val[2]);
        pack_vec_c.val[3] = vmulq_u32(pack_vec_c.val[3], pack_vec_remain_beta.val[3]);
        vst1q_u32(p_mat_aux_c+pack_ori_ind_0, pack_vec_c.val[0]);
        vst1q_u32(p_mat_aux_c+pack_ori_ind_1, pack_vec_c.val[1]);
        vst1q_u32(p_mat_aux_c+pack_ori_ind_2, pack_vec_c.val[2]);
        vst1q_u32(p_mat_aux_c+pack_ori_ind_3, pack_vec_c.val[3]);
    }

EXIT_GEMM_CPU_ACCU_NEON_UINT32:
    ree_free(p_vec_remain_beta);
    ANDROID_GEMM_FUNC_EXIT_LOG;
}

static void gemm_cpu_neon_nn_uint32(gemm_pack_param_t *p_pack_params)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    ree_log(LOG_ERROR, "%s doesn't support yet", __func__);
    ANDROID_GEMM_FUNC_EXIT_LOG;
}

static void gemm_cpu_neon_tn_uint32(gemm_pack_param_t *p_pack_params)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    ree_log(LOG_ERROR, "%s doesn't support yet", __func__);
    ANDROID_GEMM_FUNC_EXIT_LOG;
}

static void gemm_cpu_neon_nt_uint32_v1(gemm_pack_param_t *p_pack_params)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    int m_steps = 0, n_steps = 0, k_steps = 0;
    int m_step = 0, n_step = 0, k_step = 0;
    int m_offset = 0, n_offset = 0, k_offset = 0;
    int m_dims = p_pack_params->m_dims;
    int n_dims = p_pack_params->n_dims;
    int k_dims = p_pack_params->k_dims;
    int pack_heights = p_pack_params->pack_heights;
    int pack_widths = p_pack_params->pack_widths;
    int a_ele_num = 0, b_ele_num = 0;
    int aux_a_ele_num = 0, aux_b_ele_num = 0;
    int remain_k_lane = 0;
    int n_ind_0 = 0, n_ind_1 = 0, n_ind_2 = 0, n_ind_3 = 0;
    int k_ind_0 = 0, k_ind_1 = 0, k_ind_2 = 0, k_ind_3 = 0;
    int c_ind_0 = 0, c_ind_1 = 0, c_ind_2 = 0, c_ind_3 = 0;
    uint32_t *p_mat_a = (uint32_t*)p_pack_params->mat_a;
    uint32_t *p_mat_b = (uint32_t*)p_pack_params->mat_b;
    uint32_t *p_mat_aux_a = NULL;
    uint32_t *p_mat_aux_b = NULL;
    uint32_t *p_mat_aux_c = (uint32_t*)p_pack_params->mat_c_aux;
    uint32_t *p_vec_rem_alpha = NULL;
    uint64_t *p_vec_c = NULL;

    double start = 0.0f, end = 0.0f;

    uint32x4_t pack_vec_alpha;
    uint32x4_t pack_vec_remain_alpha;
    uint32x4x4_t pack_vec_aux_a;
    uint32x4x4_t pack_vec_aux_b;
    uint32x4x4_t pack_vec_aux_c_0;
    uint32x4x4_t pack_vec_aux_c_1;
    uint32x4x4_t pack_vec_aux_c_2;
    uint32x4x4_t pack_vec_aux_c_3;

    uint64x2x4_t pack_sca_c_0;
    uint64x2x4_t pack_sca_c_1;
    uint64x2x4_t pack_sca_c_2;
    uint64x2x4_t pack_sca_c_3;

    uint64x1x4_t pack_sca_cc_0;
    uint64x1x4_t pack_sca_cc_1;
    uint64x1x4_t pack_sca_cc_2;
    uint64x1x4_t pack_sca_cc_3;

    m_steps = m_dims/pack_heights + 1;
    n_steps = n_dims/pack_heights + 1;
    k_steps = k_dims/pack_widths + 1;
    
    ree_log(LOG_DEBUG, "%s m_steps %d", __func__, m_steps);
    ree_log(LOG_DEBUG, "%s n_steps %d", __func__, n_steps);
    ree_log(LOG_DEBUG, "%s k_steps %d", __func__, k_steps);

    a_ele_num = m_dims*k_dims;
    b_ele_num = n_dims*k_dims;
    aux_a_ele_num = m_steps*k_steps*pack_heights*pack_widths;
    aux_b_ele_num = n_steps*k_steps*pack_heights*pack_widths;
    remain_k_lane = pack_widths - k_steps*pack_widths + k_dims;
    ree_log(LOG_DEBUG, "%s a_ele_num %d", __func__, a_ele_num);
    ree_log(LOG_DEBUG, "%s b_ele_num %d", __func__, b_ele_num);
    ree_log(LOG_DEBUG, "%s aux_a_ele_num %d", __func__, aux_a_ele_num);
    ree_log(LOG_DEBUG, "%s aux_b_ele_num %d", __func__, aux_b_ele_num);
    ree_log(LOG_DEBUG, "%s remian_k_lane %d", __func__, remain_k_lane);

    p_mat_aux_a = (uint32_t*)ree_malloc(sizeof(uint32_t)*aux_a_ele_num);
    p_mat_aux_b = (uint32_t*)ree_malloc(sizeof(uint32_t)*aux_b_ele_num);
    p_vec_rem_alpha = (uint32_t*)ree_malloc(sizeof(uint32_t)*pack_widths);
    p_vec_c = (uint64_t*)ree_malloc(sizeof(uint64_t)*pack_widths*pack_heights);

    if ((!p_mat_aux_a)||(!p_mat_aux_b)||(!p_vec_c)||(!p_vec_rem_alpha))
    {
        ree_log(LOG_ERROR, "%s allocates p_mat_aux_a/p_mat_aux_b/p_vec_c/p_vec_rem_alpha failed", __func__);
        goto EXIT_GEMM_CPU_NEON_NT_UINT32;
    }
    
    ree_log(LOG_DEBUG, "%s alpha %d", __func__, p_pack_params->alpha);
    ree_set(p_mat_aux_a, 0, sizeof(uint32_t)*aux_a_ele_num);
    ree_set(p_mat_aux_b, 0, sizeof(uint32_t)*aux_b_ele_num);
    ree_cpy(p_mat_aux_a, p_mat_a, sizeof(uint32_t)*a_ele_num);
    ree_cpy(p_mat_aux_b, p_mat_b, sizeof(uint32_t)*b_ele_num);
    ree_set(p_vec_rem_alpha, 0, sizeof(uint32_t)*pack_widths);
    ree_set(p_vec_c, 0, sizeof(uint64_t)*pack_widths*pack_heights);

    start = now_ns();
    for (int i = 0; i<remain_k_lane; i++)
        p_vec_rem_alpha[i] = p_pack_params->alpha;

    pack_vec_alpha = vdupq_n_u32(p_pack_params->alpha);
    pack_vec_remain_alpha = vld1q_u32(p_vec_rem_alpha);

    for (m_step = 0; m_step<m_steps; m_step++)
    {
        m_offset = m_step *pack_heights;
        // ree_log(LOG_DEBUG, "%s m_offset %d", __func__, m_offset);
        for (k_step = 0; k_step<(k_steps-1); k_step++)
        {
            k_offset = k_step * pack_widths;
            k_ind_0 = m_offset*k_dims+k_offset;
            k_ind_1 = k_dims+k_ind_0;
            k_ind_2 = k_dims+k_ind_1;
            k_ind_3 = k_dims+k_ind_2;
            // ree_log(LOG_DEBUG, "%s k_offset %d", __func__, k_offset);
            // ree_log(LOG_DEBUG, "%s k_ind_0 %d", __func__, k_ind_0);
            // ree_log(LOG_DEBUG, "%s k_ind_1 %d", __func__, k_ind_1);
            // ree_log(LOG_DEBUG, "%s k_ind_2 %d", __func__, k_ind_2);
            // ree_log(LOG_DEBUG, "%s k_ind_3 %d", __func__, k_ind_3);
            pack_vec_aux_a.val[0] = vld1q_u32(p_mat_aux_a + k_ind_0);
            pack_vec_aux_a.val[1] = vld1q_u32(p_mat_aux_a + k_ind_1);
            pack_vec_aux_a.val[2] = vld1q_u32(p_mat_aux_a + k_ind_2);
            pack_vec_aux_a.val[3] = vld1q_u32(p_mat_aux_a + k_ind_3);
            pack_vec_aux_a.val[0] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_alpha);
            pack_vec_aux_a.val[1] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_alpha);
            pack_vec_aux_a.val[2] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_alpha);
            pack_vec_aux_a.val[3] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_alpha);
            for (n_step = 0; n_step<n_steps; n_step++)
            {
                n_offset = n_step*pack_heights;
                n_ind_0 = n_offset*k_dims+k_offset;
                n_ind_1 = n_ind_0+k_dims;
                n_ind_2 = n_ind_1+k_dims;
                n_ind_3 = n_ind_2+k_dims;
                c_ind_0 = m_offset*n_dims+n_offset;
                c_ind_1 = n_dims+c_ind_0;
                c_ind_2 = n_dims+c_ind_1;
                c_ind_3 = n_dims+c_ind_2;
                // ree_log(LOG_DEBUG, "%s n_offset %d", __func__, n_offset);
                // ree_log(LOG_DEBUG, "%s n_ind_0 %d", __func__, n_ind_0);
                // ree_log(LOG_DEBUG, "%s n_ind_1 %d", __func__, n_ind_1);
                // ree_log(LOG_DEBUG, "%s n_ind_2 %d", __func__, n_ind_2);
                // ree_log(LOG_DEBUG, "%s n_ind_3 %d", __func__, n_ind_3);
                // ree_log(LOG_DEBUG, "%s c_ind_0 %d", __func__, c_ind_0);
                // ree_log(LOG_DEBUG, "%s c_ind_1 %d", __func__, c_ind_1);
                // ree_log(LOG_DEBUG, "%s c_ind_2 %d", __func__, c_ind_2);
                // ree_log(LOG_DEBUG, "%s c_ind_3 %d", __func__, c_ind_3);
                pack_vec_aux_b.val[0] = vld1q_u32(p_mat_aux_b + n_ind_0);
                pack_vec_aux_b.val[1] = vld1q_u32(p_mat_aux_b + n_ind_1);
                pack_vec_aux_b.val[2] = vld1q_u32(p_mat_aux_b + n_ind_2);
                pack_vec_aux_b.val[3] = vld1q_u32(p_mat_aux_b + n_ind_3);
                
                pack_vec_aux_c_0.val[0] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_aux_b.val[0]);
                pack_vec_aux_c_0.val[1] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_aux_b.val[1]);
                pack_vec_aux_c_0.val[2] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_aux_b.val[2]);
                pack_vec_aux_c_0.val[3] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_aux_b.val[3]);

                pack_vec_aux_c_1.val[0] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_aux_b.val[0]);
                pack_vec_aux_c_1.val[1] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_aux_b.val[1]);
                pack_vec_aux_c_1.val[2] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_aux_b.val[2]);
                pack_vec_aux_c_1.val[3] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_aux_b.val[3]);

                pack_vec_aux_c_2.val[0] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_aux_b.val[0]);
                pack_vec_aux_c_2.val[1] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_aux_b.val[1]);
                pack_vec_aux_c_2.val[2] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_aux_b.val[2]);
                pack_vec_aux_c_2.val[3] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_aux_b.val[3]);

                pack_vec_aux_c_3.val[0] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_aux_b.val[0]);
                pack_vec_aux_c_3.val[1] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_aux_b.val[1]);
                pack_vec_aux_c_3.val[2] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_aux_b.val[2]);
                pack_vec_aux_c_3.val[3] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_aux_b.val[3]);

                pack_sca_c_0.val[0] = vpaddlq_u32(pack_vec_aux_c_0.val[0]);
                pack_sca_c_0.val[1] = vpaddlq_u32(pack_vec_aux_c_0.val[1]);
                pack_sca_c_0.val[2] = vpaddlq_u32(pack_vec_aux_c_0.val[2]);
                pack_sca_c_0.val[3] = vpaddlq_u32(pack_vec_aux_c_0.val[3]);

                pack_sca_c_1.val[0] = vpaddlq_u32(pack_vec_aux_c_1.val[0]);
                pack_sca_c_1.val[1] = vpaddlq_u32(pack_vec_aux_c_1.val[1]);
                pack_sca_c_1.val[2] = vpaddlq_u32(pack_vec_aux_c_1.val[2]);
                pack_sca_c_1.val[3] = vpaddlq_u32(pack_vec_aux_c_1.val[3]);

                pack_sca_c_2.val[0] = vpaddlq_u32(pack_vec_aux_c_2.val[0]);
                pack_sca_c_2.val[1] = vpaddlq_u32(pack_vec_aux_c_2.val[1]);
                pack_sca_c_2.val[2] = vpaddlq_u32(pack_vec_aux_c_2.val[2]);
                pack_sca_c_2.val[3] = vpaddlq_u32(pack_vec_aux_c_2.val[3]);

                pack_sca_c_3.val[0] = vpaddlq_u32(pack_vec_aux_c_3.val[0]);
                pack_sca_c_3.val[1] = vpaddlq_u32(pack_vec_aux_c_3.val[1]);
                pack_sca_c_3.val[2] = vpaddlq_u32(pack_vec_aux_c_3.val[2]);
                pack_sca_c_3.val[3] = vpaddlq_u32(pack_vec_aux_c_3.val[3]);

                pack_sca_cc_0.val[0] = vadd_u64(vget_high_u64(pack_sca_c_0.val[0]),
                                                vget_low_u64(pack_sca_c_0.val[0]));
                pack_sca_cc_0.val[1] = vadd_u64(vget_high_u64(pack_sca_c_0.val[1]),
                                                vget_low_u64(pack_sca_c_0.val[1]));
                pack_sca_cc_0.val[2] = vadd_u64(vget_high_u64(pack_sca_c_0.val[2]),
                                                vget_low_u64(pack_sca_c_0.val[2]));
                pack_sca_cc_0.val[3] = vadd_u64(vget_high_u64(pack_sca_c_0.val[3]),
                                                vget_low_u64(pack_sca_c_0.val[3]));

                pack_sca_cc_1.val[0] = vadd_u64(vget_high_u64(pack_sca_c_1.val[0]),
                                                vget_low_u64(pack_sca_c_1.val[0]));
                pack_sca_cc_1.val[1] = vadd_u64(vget_high_u64(pack_sca_c_1.val[1]),
                                                vget_low_u64(pack_sca_c_1.val[1]));
                pack_sca_cc_1.val[2] = vadd_u64(vget_high_u64(pack_sca_c_1.val[2]),
                                                vget_low_u64(pack_sca_c_1.val[2]));
                pack_sca_cc_1.val[3] = vadd_u64(vget_high_u64(pack_sca_c_1.val[3]),
                                                vget_low_u64(pack_sca_c_1.val[3]));

                pack_sca_cc_2.val[0] = vadd_u64(vget_high_u64(pack_sca_c_2.val[0]),
                                                vget_low_u64(pack_sca_c_2.val[0]));
                pack_sca_cc_2.val[1] = vadd_u64(vget_high_u64(pack_sca_c_2.val[1]),
                                                vget_low_u64(pack_sca_c_2.val[1]));
                pack_sca_cc_2.val[2] = vadd_u64(vget_high_u64(pack_sca_c_2.val[2]),
                                                vget_low_u64(pack_sca_c_2.val[2]));
                pack_sca_cc_2.val[3] = vadd_u64(vget_high_u64(pack_sca_c_2.val[3]),
                                                vget_low_u64(pack_sca_c_2.val[3]));

                pack_sca_cc_3.val[0] = vadd_u64(vget_high_u64(pack_sca_c_3.val[0]),
                                                vget_low_u64(pack_sca_c_3.val[0]));
                pack_sca_cc_3.val[1] = vadd_u64(vget_high_u64(pack_sca_c_3.val[1]),
                                                vget_low_u64(pack_sca_c_3.val[1]));
                pack_sca_cc_3.val[2] = vadd_u64(vget_high_u64(pack_sca_c_3.val[2]),
                                                vget_low_u64(pack_sca_c_3.val[2]));
                pack_sca_cc_3.val[3] = vadd_u64(vget_high_u64(pack_sca_c_3.val[3]),
                                                vget_low_u64(pack_sca_c_3.val[3]));

                vst1_u64(p_vec_c, pack_sca_cc_0.val[0]);
                vst1_u64(p_vec_c+1, pack_sca_cc_0.val[1]);
                vst1_u64(p_vec_c+2, pack_sca_cc_0.val[2]);
                vst1_u64(p_vec_c+3, pack_sca_cc_0.val[3]);
                vst1_u64(p_vec_c+pack_widths, pack_sca_cc_1.val[0]);
                vst1_u64(p_vec_c+pack_widths+1, pack_sca_cc_1.val[1]);
                vst1_u64(p_vec_c+pack_widths+2, pack_sca_cc_1.val[2]);
                vst1_u64(p_vec_c+pack_widths+3, pack_sca_cc_1.val[3]);
                vst1_u64(p_vec_c+pack_widths*2, pack_sca_cc_2.val[0]);
                vst1_u64(p_vec_c+pack_widths*2+1, pack_sca_cc_2.val[1]);
                vst1_u64(p_vec_c+pack_widths*2+2, pack_sca_cc_2.val[2]);
                vst1_u64(p_vec_c+pack_widths*2+3, pack_sca_cc_2.val[3]);
                vst1_u64(p_vec_c+pack_widths*3, pack_sca_cc_3.val[0]);
                vst1_u64(p_vec_c+pack_widths*3+1, pack_sca_cc_3.val[1]);
                vst1_u64(p_vec_c+pack_widths*3+2, pack_sca_cc_3.val[2]);
                vst1_u64(p_vec_c+pack_widths*3+3, pack_sca_cc_3.val[3]);

                *(p_mat_aux_c+c_ind_0) += (uint32_t)p_vec_c[0];
                *(p_mat_aux_c+c_ind_0+1) += (uint32_t)p_vec_c[1];
                *(p_mat_aux_c+c_ind_0+2) += (uint32_t)p_vec_c[2];
                *(p_mat_aux_c+c_ind_0+3) += (uint32_t)p_vec_c[3];

                *(p_mat_aux_c+c_ind_1) += (uint32_t)p_vec_c[pack_widths];
                *(p_mat_aux_c+c_ind_1+1) += (uint32_t)p_vec_c[pack_widths+1];
                *(p_mat_aux_c+c_ind_1+2) += (uint32_t)p_vec_c[pack_widths+2];
                *(p_mat_aux_c+c_ind_1+3) += (uint32_t)p_vec_c[pack_widths+3];

                *(p_mat_aux_c+c_ind_2) += (uint32_t)p_vec_c[pack_widths*2];
                *(p_mat_aux_c+c_ind_2+1) += (uint32_t)p_vec_c[pack_widths*2+1];
                *(p_mat_aux_c+c_ind_2+2) += (uint32_t)p_vec_c[pack_widths*2+2];
                *(p_mat_aux_c+c_ind_2+3) += (uint32_t)p_vec_c[pack_widths*2+3];

                *(p_mat_aux_c+c_ind_3) += (uint32_t)p_vec_c[pack_widths*3];
                *(p_mat_aux_c+c_ind_3+1) += (uint32_t)p_vec_c[pack_widths*3+1];
                *(p_mat_aux_c+c_ind_3+2) += (uint32_t)p_vec_c[pack_widths*3+2];
                *(p_mat_aux_c+c_ind_3+3) += (uint32_t)p_vec_c[pack_widths*3+3];
            }
        }

        for (n_step = 0; n_step<n_steps; n_step++)
        {
            k_offset = (k_steps-1)*pack_widths;
            k_ind_0 = m_offset*k_dims+k_offset;
            k_ind_1 = k_dims+k_ind_0;
            k_ind_2 = k_dims+k_ind_1;
            k_ind_3 = k_dims+k_ind_2;

            n_offset = n_step*pack_heights;
            n_ind_0 = n_offset*k_dims+k_offset;
            n_ind_1 = k_dims+n_ind_0;
            n_ind_2 = k_dims+n_ind_1;
            n_ind_3 = k_dims+n_ind_2;
            c_ind_0 = m_offset*n_dims+n_offset;
            c_ind_1 = n_dims+c_ind_0;
            c_ind_2 = n_dims+c_ind_1;
            c_ind_3 = n_dims+c_ind_2;
            // ree_log(LOG_DEBUG, "%s n_offset %d", __func__, n_offset);
            // ree_log(LOG_DEBUG, "%s n_ind_0 %d", __func__, n_ind_0);
            // ree_log(LOG_DEBUG, "%s n_ind_1 %d", __func__, n_ind_1);
            // ree_log(LOG_DEBUG, "%s n_ind_2 %d", __func__, n_ind_2);
            // ree_log(LOG_DEBUG, "%s n_ind_3 %d", __func__, n_ind_3);
            // ree_log(LOG_DEBUG, "%s c_ind_0 %d", __func__, c_ind_0);
            // ree_log(LOG_DEBUG, "%s c_ind_1 %d", __func__, c_ind_1);
            // ree_log(LOG_DEBUG, "%s c_ind_2 %d", __func__, c_ind_2);
            // ree_log(LOG_DEBUG, "%s c_ind_3 %d", __func__, c_ind_3);
            pack_vec_aux_a.val[0] = vld1q_u32(p_mat_aux_a + k_ind_0);
            pack_vec_aux_a.val[1] = vld1q_u32(p_mat_aux_a + k_ind_1);
            pack_vec_aux_a.val[2] = vld1q_u32(p_mat_aux_a + k_ind_2);
            pack_vec_aux_a.val[3] = vld1q_u32(p_mat_aux_a + k_ind_3);
            
            pack_vec_aux_a.val[0] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_remain_alpha);
            pack_vec_aux_a.val[1] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_remain_alpha);
            pack_vec_aux_a.val[2] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_remain_alpha);
            pack_vec_aux_a.val[3] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_remain_alpha);
            
            pack_vec_aux_b.val[0] = vld1q_u32(p_mat_aux_b + n_ind_0);
            pack_vec_aux_b.val[1] = vld1q_u32(p_mat_aux_b + n_ind_1);
            pack_vec_aux_b.val[2] = vld1q_u32(p_mat_aux_b + n_ind_2);
            pack_vec_aux_b.val[3] = vld1q_u32(p_mat_aux_b + n_ind_3);

            pack_vec_aux_c_0.val[0] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_aux_b.val[0]);
            pack_vec_aux_c_0.val[1] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_aux_b.val[1]);
            pack_vec_aux_c_0.val[2] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_aux_b.val[2]);
            pack_vec_aux_c_0.val[3] = vmulq_u32(pack_vec_aux_a.val[0], pack_vec_aux_b.val[3]);

            pack_vec_aux_c_1.val[0] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_aux_b.val[0]);
            pack_vec_aux_c_1.val[1] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_aux_b.val[1]);
            pack_vec_aux_c_1.val[2] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_aux_b.val[2]);
            pack_vec_aux_c_1.val[3] = vmulq_u32(pack_vec_aux_a.val[1], pack_vec_aux_b.val[3]);

            pack_vec_aux_c_2.val[0] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_aux_b.val[0]);
            pack_vec_aux_c_2.val[1] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_aux_b.val[1]);
            pack_vec_aux_c_2.val[2] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_aux_b.val[2]);
            pack_vec_aux_c_2.val[3] = vmulq_u32(pack_vec_aux_a.val[2], pack_vec_aux_b.val[3]);

            pack_vec_aux_c_3.val[0] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_aux_b.val[0]);
            pack_vec_aux_c_3.val[1] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_aux_b.val[1]);
            pack_vec_aux_c_3.val[2] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_aux_b.val[2]);
            pack_vec_aux_c_3.val[3] = vmulq_u32(pack_vec_aux_a.val[3], pack_vec_aux_b.val[3]);

            pack_sca_c_0.val[0] = vpaddlq_u32(pack_vec_aux_c_0.val[0]);
            pack_sca_c_0.val[1] = vpaddlq_u32(pack_vec_aux_c_0.val[1]);
            pack_sca_c_0.val[2] = vpaddlq_u32(pack_vec_aux_c_0.val[2]);
            pack_sca_c_0.val[3] = vpaddlq_u32(pack_vec_aux_c_0.val[3]);

            pack_sca_c_1.val[0] = vpaddlq_u32(pack_vec_aux_c_1.val[0]);
            pack_sca_c_1.val[1] = vpaddlq_u32(pack_vec_aux_c_1.val[1]);
            pack_sca_c_1.val[2] = vpaddlq_u32(pack_vec_aux_c_1.val[2]);
            pack_sca_c_1.val[3] = vpaddlq_u32(pack_vec_aux_c_1.val[3]);

            pack_sca_c_2.val[0] = vpaddlq_u32(pack_vec_aux_c_2.val[0]);
            pack_sca_c_2.val[1] = vpaddlq_u32(pack_vec_aux_c_2.val[1]);
            pack_sca_c_2.val[2] = vpaddlq_u32(pack_vec_aux_c_2.val[2]);
            pack_sca_c_2.val[3] = vpaddlq_u32(pack_vec_aux_c_2.val[3]);

            pack_sca_c_3.val[0] = vpaddlq_u32(pack_vec_aux_c_3.val[0]);
            pack_sca_c_3.val[1] = vpaddlq_u32(pack_vec_aux_c_3.val[1]);
            pack_sca_c_3.val[2] = vpaddlq_u32(pack_vec_aux_c_3.val[2]);
            pack_sca_c_3.val[3] = vpaddlq_u32(pack_vec_aux_c_3.val[3]);

            pack_sca_cc_0.val[0] = vadd_u64(vget_high_u64(pack_sca_c_0.val[0]),
                                            vget_low_u64(pack_sca_c_0.val[0]));
            pack_sca_cc_0.val[1] = vadd_u64(vget_high_u64(pack_sca_c_0.val[1]),
                                            vget_low_u64(pack_sca_c_0.val[1]));
            pack_sca_cc_0.val[2] = vadd_u64(vget_high_u64(pack_sca_c_0.val[2]),
                                            vget_low_u64(pack_sca_c_0.val[2]));
            pack_sca_cc_0.val[3] = vadd_u64(vget_high_u64(pack_sca_c_0.val[3]),
                                            vget_low_u64(pack_sca_c_0.val[3]));

            pack_sca_cc_1.val[0] = vadd_u64(vget_high_u64(pack_sca_c_1.val[0]),
                                            vget_low_u64(pack_sca_c_1.val[0]));
            pack_sca_cc_1.val[1] = vadd_u64(vget_high_u64(pack_sca_c_1.val[1]),
                                            vget_low_u64(pack_sca_c_1.val[1]));
            pack_sca_cc_1.val[2] = vadd_u64(vget_high_u64(pack_sca_c_1.val[2]),
                                            vget_low_u64(pack_sca_c_1.val[2]));
            pack_sca_cc_1.val[3] = vadd_u64(vget_high_u64(pack_sca_c_1.val[3]),
                                            vget_low_u64(pack_sca_c_1.val[3]));

            pack_sca_cc_2.val[0] = vadd_u64(vget_high_u64(pack_sca_c_2.val[0]),
                                            vget_low_u64(pack_sca_c_2.val[0]));
            pack_sca_cc_2.val[1] = vadd_u64(vget_high_u64(pack_sca_c_2.val[1]),
                                            vget_low_u64(pack_sca_c_2.val[1]));
            pack_sca_cc_2.val[2] = vadd_u64(vget_high_u64(pack_sca_c_2.val[2]),
                                            vget_low_u64(pack_sca_c_2.val[2]));
            pack_sca_cc_2.val[3] = vadd_u64(vget_high_u64(pack_sca_c_2.val[3]),
                                            vget_low_u64(pack_sca_c_2.val[3]));

            pack_sca_cc_3.val[0] = vadd_u64(vget_high_u64(pack_sca_c_3.val[0]),
                                            vget_low_u64(pack_sca_c_3.val[0]));
            pack_sca_cc_3.val[1] = vadd_u64(vget_high_u64(pack_sca_c_3.val[1]),
                                            vget_low_u64(pack_sca_c_3.val[1]));
            pack_sca_cc_3.val[2] = vadd_u64(vget_high_u64(pack_sca_c_3.val[2]),
                                            vget_low_u64(pack_sca_c_3.val[2]));
            pack_sca_cc_3.val[3] = vadd_u64(vget_high_u64(pack_sca_c_3.val[3]),
                                            vget_low_u64(pack_sca_c_3.val[3]));


            vst1_u64(p_vec_c, pack_sca_cc_0.val[0]);
            vst1_u64(p_vec_c+1, pack_sca_cc_0.val[1]);
            vst1_u64(p_vec_c+2, pack_sca_cc_0.val[2]);
            vst1_u64(p_vec_c+3, pack_sca_cc_0.val[3]);
            vst1_u64(p_vec_c+pack_widths, pack_sca_cc_1.val[0]);
            vst1_u64(p_vec_c+pack_widths+1, pack_sca_cc_1.val[1]);
            vst1_u64(p_vec_c+pack_widths+2, pack_sca_cc_1.val[2]);
            vst1_u64(p_vec_c+pack_widths+3, pack_sca_cc_1.val[3]);
            vst1_u64(p_vec_c+pack_widths*2, pack_sca_cc_2.val[0]);
            vst1_u64(p_vec_c+pack_widths*2+1, pack_sca_cc_2.val[1]);
            vst1_u64(p_vec_c+pack_widths*2+2, pack_sca_cc_2.val[2]);
            vst1_u64(p_vec_c+pack_widths*2+3, pack_sca_cc_2.val[3]);
            vst1_u64(p_vec_c+pack_widths*3, pack_sca_cc_3.val[0]);
            vst1_u64(p_vec_c+pack_widths*3+1, pack_sca_cc_3.val[1]);
            vst1_u64(p_vec_c+pack_widths*3+2, pack_sca_cc_3.val[2]);
            vst1_u64(p_vec_c+pack_widths*3+3, pack_sca_cc_3.val[3]);

            
            *(p_mat_aux_c+c_ind_0) += (uint32_t)(p_vec_c[0]);
            *(p_mat_aux_c+c_ind_0+1) += (uint32_t)(p_vec_c[1]);
            *(p_mat_aux_c+c_ind_0+2) += (uint32_t)(p_vec_c[2]);
            *(p_mat_aux_c+c_ind_0+3) += (uint32_t)(p_vec_c[3]);

            *(p_mat_aux_c+c_ind_1) += (uint32_t)(p_vec_c[pack_widths]);
            *(p_mat_aux_c+c_ind_1+1) += (uint32_t)(p_vec_c[pack_widths+1]);
            *(p_mat_aux_c+c_ind_1+2) += (uint32_t)(p_vec_c[pack_widths+2]);
            *(p_mat_aux_c+c_ind_1+3) += (uint32_t)(p_vec_c[pack_widths+3]);

            *(p_mat_aux_c+c_ind_2) += (uint32_t)(p_vec_c[pack_widths*2]);
            *(p_mat_aux_c+c_ind_2+1) += (uint32_t)(p_vec_c[pack_widths*2+1]);
            *(p_mat_aux_c+c_ind_2+2) += (uint32_t)(p_vec_c[pack_widths*2+2]);
            *(p_mat_aux_c+c_ind_2+3) += (uint32_t)(p_vec_c[pack_widths*2+3]);

            *(p_mat_aux_c+c_ind_3) += (uint32_t)p_vec_c[pack_widths*3];
            *(p_mat_aux_c+c_ind_3+1) += (uint32_t)p_vec_c[pack_widths*3+1];
            *(p_mat_aux_c+c_ind_3+2) += (uint32_t)p_vec_c[pack_widths*3+2];
            *(p_mat_aux_c+c_ind_3+3) += (uint32_t)p_vec_c[pack_widths*3+3];
        }
    }

    end = now_ns();
    ree_log(LOG_DEBUG, "%s executes times %f", __func__, end - start);

EXIT_GEMM_CPU_NEON_NT_UINT32:
    ree_free(p_mat_aux_a);
    ree_free(p_mat_aux_b);
    ree_free(p_vec_rem_alpha);
    ree_free(p_vec_c);
    ANDROID_GEMM_FUNC_EXIT_LOG;
}


static void gemm_cpu_neon_nt_uint32_v2(gemm_pack_param_t *p_pack_params)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    int m_steps = 0, n_steps = 0, k_steps = 0;
    int m_step = 0, n_step = 0, k_step = 0;
    int m_offset = 0, n_offset = 0, k_offset = 0;
    int m_dims = p_pack_params->m_dims;
    int n_dims = p_pack_params->n_dims;
    int k_dims = p_pack_params->k_dims;
    int pack_heights = p_pack_params->pack_heights;
    int pack_widths = p_pack_params->pack_widths;
    int a_ele_num = 0, b_ele_num = 0;
    int aux_a_ele_num = 0, aux_b_ele_num = 0;
    int remain_k_lane = 0;
    int n_ind_0 = 0, n_ind_1 = 0, n_ind_2 = 0, n_ind_3 = 0;
    int k_ind_0 = 0, k_ind_1 = 0, k_ind_2 = 0, k_ind_3 = 0;
    int c_ind_0 = 0, c_ind_1 = 0, c_ind_2 = 0, c_ind_3 = 0;
    uint32_t *p_mat_a = (uint32_t*)p_pack_params->mat_a;
    uint32_t *p_mat_b = (uint32_t*)p_pack_params->mat_b;
    uint32_t *p_mat_aux_a = NULL;
    uint32_t *p_mat_aux_b = NULL;
    uint32_t *p_mat_aux_c = (uint32_t*)p_pack_params->mat_c_aux;
    uint32_t *p_mat_pack_b = NULL;
    uint64_t *p_vec_c = NULL;

    double start = 0.0f, end = 0.0f;

    uint32x4x4_t pack_vec_aux_a;
    uint32x4x4_t pack_vec_aux_b;
    uint32x4x4_t pack_vec_aux_c;

    m_steps = m_dims/pack_heights + 1;
    n_steps = n_dims/pack_heights + 1;
    k_steps = k_dims/pack_widths + 1;
    
    ree_log(LOG_DEBUG, "%s m_steps %d", __func__, m_steps);
    ree_log(LOG_DEBUG, "%s n_steps %d", __func__, n_steps);
    ree_log(LOG_DEBUG, "%s k_steps %d", __func__, k_steps);

    a_ele_num = m_dims*k_dims;
    b_ele_num = n_dims*k_dims;
    aux_a_ele_num = m_steps*k_steps*pack_heights*pack_widths;
    aux_b_ele_num = n_steps*k_steps*pack_heights*pack_widths;
    remain_k_lane = pack_widths - k_steps*pack_widths + k_dims;
    // ree_log(LOG_DEBUG, "%s a_ele_num %d", __func__, a_ele_num);
    // ree_log(LOG_DEBUG, "%s b_ele_num %d", __func__, b_ele_num);
    // ree_log(LOG_DEBUG, "%s aux_a_ele_num %d", __func__, aux_a_ele_num);
    // ree_log(LOG_DEBUG, "%s aux_b_ele_num %d", __func__, aux_b_ele_num);
    // ree_log(LOG_DEBUG, "%s remian_k_lane %d", __func__, remain_k_lane);

    p_mat_aux_a = (uint32_t*)ree_malloc(sizeof(uint32_t)*aux_a_ele_num);
    p_mat_aux_b = (uint32_t*)ree_malloc(sizeof(uint32_t)*aux_b_ele_num);
    p_mat_pack_b = (uint32_t*)ree_malloc(sizeof(uint32_t)*pack_heights*pack_widths);
    p_vec_c = (uint64_t*)ree_malloc(sizeof(uint64_t)*pack_widths*pack_heights);

    if ((!p_mat_aux_a)||(!p_mat_aux_b)||(!p_vec_c)||(!p_mat_pack_b))
    {
        ree_log(LOG_ERROR, "%s allocates p_mat_aux_a/p_mat_aux_b/p_vec_c/p_mat_pack_b failed", __func__);
        goto EXIT_GEMM_CPU_NEON_NT_UINT32;
    }
    
    // ree_log(LOG_DEBUG, "%s alpha %d", __func__, p_pack_params->alpha);
    ree_set(p_mat_aux_a, 0, sizeof(uint32_t)*aux_a_ele_num);
    ree_set(p_mat_aux_b, 0, sizeof(uint32_t)*aux_b_ele_num);
    ree_cpy(p_mat_aux_a, p_mat_a, sizeof(uint32_t)*a_ele_num);
    ree_cpy(p_mat_aux_b, p_mat_b, sizeof(uint32_t)*b_ele_num);
    // ree_set(p_mat_pack_b, 0, sizeof(uint32_t)*pack_heights*pack_widths);

    start = now_ns();

    for (m_step = 0; m_step<m_steps; m_step++)
    {
        m_offset = m_step *pack_heights;
        // ree_log(LOG_DEBUG, "%s m_offset %d", __func__, m_offset);
        for (k_step = 0; k_step<(k_steps-1); k_step++)
        {
            k_offset = k_step*pack_widths;
            k_ind_0 = m_offset*k_dims+k_offset;
            k_ind_1 = k_dims+k_ind_0;
            k_ind_2 = k_dims+k_ind_1;
            k_ind_3 = k_dims+k_ind_2;
            // ree_log(LOG_DEBUG, "%s k_offset %d", __func__, k_offset);
            // ree_log(LOG_DEBUG, "%s k_ind_0 %d", __func__, k_ind_0);
            // ree_log(LOG_DEBUG, "%s k_ind_1 %d", __func__, k_ind_1);
            // ree_log(LOG_DEBUG, "%s k_ind_2 %d", __func__, k_ind_2);
            // ree_log(LOG_DEBUG, "%s k_ind_3 %d", __func__, k_ind_3);
            pack_vec_aux_a.val[0] = vld1q_u32(p_mat_aux_a+k_ind_0);
            pack_vec_aux_a.val[1] = vld1q_u32(p_mat_aux_a+k_ind_1);
            pack_vec_aux_a.val[2] = vld1q_u32(p_mat_aux_a+k_ind_2);
            pack_vec_aux_a.val[3] = vld1q_u32(p_mat_aux_a+k_ind_3);
            for (n_step = 0; n_step<n_steps; n_step++)
            {
                n_offset = n_step*pack_heights;
                n_ind_0 = n_offset*k_dims+k_offset;
                n_ind_1 = k_dims+n_ind_0;
                n_ind_2 = k_dims+n_ind_1;
                n_ind_3 = k_dims+n_ind_2;
                c_ind_0 = m_offset*n_dims+n_offset;
                c_ind_1 = n_dims+c_ind_0;
                c_ind_2 = n_dims+c_ind_1;
                c_ind_3 = n_dims+c_ind_2;
                // ree_log(LOG_DEBUG, "%s n_offset %d", __func__, n_offset);
                // ree_log(LOG_DEBUG, "%s n_ind_0 %d", __func__, n_ind_0);
                // ree_log(LOG_DEBUG, "%s n_ind_1 %d", __func__, n_ind_1);
                // ree_log(LOG_DEBUG, "%s n_ind_2 %d", __func__, n_ind_2);
                // ree_log(LOG_DEBUG, "%s n_ind_3 %d", __func__, n_ind_3);
                // ree_log(LOG_DEBUG, "%s c_ind_0 %d", __func__, c_ind_0);
                // ree_log(LOG_DEBUG, "%s c_ind_1 %d", __func__, c_ind_1);
                // ree_log(LOG_DEBUG, "%s c_ind_2 %d", __func__, c_ind_2);
                // ree_log(LOG_DEBUG, "%s c_ind_3 %d", __func__, c_ind_3);
                ree_cpy(p_mat_pack_b, p_mat_aux_b+n_ind_0, sizeof(uint32_t)*pack_widths);
                ree_cpy(p_mat_pack_b+pack_widths, p_mat_aux_b+n_ind_1, sizeof(uint32_t)*pack_widths);
                ree_cpy(p_mat_pack_b+pack_widths*2, p_mat_aux_b+n_ind_2, sizeof(uint32_t)*pack_widths);
                ree_cpy(p_mat_pack_b+pack_widths*3, p_mat_aux_b+n_ind_3, sizeof(uint32_t)*pack_widths);
                pack_vec_aux_b = vld4q_u32(p_mat_pack_b);
                pack_vec_aux_c.val[0] = vld1q_u32(p_mat_aux_c+c_ind_0);
                pack_vec_aux_c.val[1] = vld1q_u32(p_mat_aux_c+c_ind_1);
                pack_vec_aux_c.val[2] = vld1q_u32(p_mat_aux_c+c_ind_2);
                pack_vec_aux_c.val[3] = vld1q_u32(p_mat_aux_c+c_ind_3);
                pack_vec_aux_b.val[0] = vmulq_n_u32(pack_vec_aux_b.val[0], p_pack_params->alpha);
                pack_vec_aux_b.val[1] = vmulq_n_u32(pack_vec_aux_b.val[1], p_pack_params->alpha);
                pack_vec_aux_b.val[2] = vmulq_n_u32(pack_vec_aux_b.val[2], p_pack_params->alpha);
                pack_vec_aux_b.val[3] = vmulq_n_u32(pack_vec_aux_b.val[3], p_pack_params->alpha);

                pack_vec_aux_c.val[0] = vmlaq_lane_u32(pack_vec_aux_c.val[0],
                                                       pack_vec_aux_b.val[0], 
                                                       vget_low_u32(pack_vec_aux_a.val[0]), 0);
                pack_vec_aux_c.val[0] = vmlaq_lane_u32(pack_vec_aux_c.val[0],
                                                       pack_vec_aux_b.val[1], 
                                                       vget_low_u32(pack_vec_aux_a.val[0]), 1);
                pack_vec_aux_c.val[0] = vmlaq_lane_u32(pack_vec_aux_c.val[0],
                                                       pack_vec_aux_b.val[2], 
                                                       vget_high_u32(pack_vec_aux_a.val[0]), 0);
                pack_vec_aux_c.val[0] = vmlaq_lane_u32(pack_vec_aux_c.val[0],
                                                       pack_vec_aux_b.val[3], 
                                                       vget_high_u32(pack_vec_aux_a.val[0]), 1);

                pack_vec_aux_c.val[1] = vmlaq_lane_u32(pack_vec_aux_c.val[1],
                                                       pack_vec_aux_b.val[0], 
                                                       vget_low_u32(pack_vec_aux_a.val[1]), 0);
                pack_vec_aux_c.val[1] = vmlaq_lane_u32(pack_vec_aux_c.val[1],
                                                       pack_vec_aux_b.val[1], 
                                                       vget_low_u32(pack_vec_aux_a.val[1]), 1);
                pack_vec_aux_c.val[1] = vmlaq_lane_u32(pack_vec_aux_c.val[1],
                                                       pack_vec_aux_b.val[2], 
                                                       vget_high_u32(pack_vec_aux_a.val[1]), 0);
                pack_vec_aux_c.val[1] = vmlaq_lane_u32(pack_vec_aux_c.val[1],
                                                       pack_vec_aux_b.val[3], 
                                                       vget_high_u32(pack_vec_aux_a.val[1]), 1);

                pack_vec_aux_c.val[2] = vmlaq_lane_u32(pack_vec_aux_c.val[2],
                                                       pack_vec_aux_b.val[0], 
                                                       vget_low_u32(pack_vec_aux_a.val[2]), 0);
                pack_vec_aux_c.val[2] = vmlaq_lane_u32(pack_vec_aux_c.val[2],
                                                       pack_vec_aux_b.val[1], 
                                                       vget_low_u32(pack_vec_aux_a.val[2]), 1);
                pack_vec_aux_c.val[2] = vmlaq_lane_u32(pack_vec_aux_c.val[2],
                                                       pack_vec_aux_b.val[2], 
                                                       vget_high_u32(pack_vec_aux_a.val[2]), 0);
                pack_vec_aux_c.val[2] = vmlaq_lane_u32(pack_vec_aux_c.val[2],
                                                       pack_vec_aux_b.val[3], 
                                                       vget_high_u32(pack_vec_aux_a.val[2]), 1);

                pack_vec_aux_c.val[3] = vmulq_lane_u32(pack_vec_aux_c.val[3], 
                                                       vget_low_u32(pack_vec_aux_a.val[3]), 0);
                pack_vec_aux_c.val[3] = vmlaq_lane_u32(pack_vec_aux_c.val[3],
                                                       pack_vec_aux_b.val[1], 
                                                       vget_low_u32(pack_vec_aux_a.val[3]), 1);
                pack_vec_aux_c.val[3] = vmlaq_lane_u32(pack_vec_aux_c.val[3],
                                                       pack_vec_aux_b.val[2], 
                                                       vget_high_u32(pack_vec_aux_a.val[3]), 0);
                pack_vec_aux_c.val[3] = vmlaq_lane_u32(pack_vec_aux_c.val[3],
                                                       pack_vec_aux_b.val[3], 
                                                       vget_high_u32(pack_vec_aux_a.val[3]), 1);
                
                vst1q_u32(p_mat_aux_c+c_ind_0, pack_vec_aux_c.val[0]);
                vst1q_u32(p_mat_aux_c+c_ind_1, pack_vec_aux_c.val[1]);
                vst1q_u32(p_mat_aux_c+c_ind_2, pack_vec_aux_c.val[2]);
                vst1q_u32(p_mat_aux_c+c_ind_3, pack_vec_aux_c.val[3]);
            }
        }

        k_offset = (k_steps-1)*pack_widths;
        k_ind_0 = m_offset*k_dims+k_offset;
        k_ind_1 = k_dims+k_ind_0;
        k_ind_2 = k_dims+k_ind_1;
        k_ind_3 = k_dims+k_ind_2;
        pack_vec_aux_a.val[0] = vld1q_u32(p_mat_aux_a+k_ind_0);
        pack_vec_aux_a.val[1] = vld1q_u32(p_mat_aux_a+k_ind_1);
        pack_vec_aux_a.val[2] = vld1q_u32(p_mat_aux_a+k_ind_2);
        pack_vec_aux_a.val[3] = vld1q_u32(p_mat_aux_a+k_ind_3);

        for (n_step = 0; n_step<n_steps; n_step++)
        {
            n_offset = n_step*pack_heights;
            n_ind_0 = n_offset*k_dims+k_offset;
            n_ind_1 = k_dims+n_ind_0;
            n_ind_2 = k_dims+n_ind_1;
            n_ind_3 = k_dims+n_ind_2;
            c_ind_0 = m_offset*n_dims+n_offset;
            c_ind_1 = n_dims+c_ind_0;
            c_ind_2 = n_dims+c_ind_1;
            c_ind_3 = n_dims+c_ind_2;
            // ree_log(LOG_DEBUG, "%s n_offset %d", __func__, n_offset);
            // ree_log(LOG_DEBUG, "%s n_ind_0 %d", __func__, n_ind_0);
            // ree_log(LOG_DEBUG, "%s n_ind_1 %d", __func__, n_ind_1);
            // ree_log(LOG_DEBUG, "%s n_ind_2 %d", __func__, n_ind_2);
            // ree_log(LOG_DEBUG, "%s n_ind_3 %d", __func__, n_ind_3);
            // ree_log(LOG_DEBUG, "%s c_ind_0 %d", __func__, c_ind_0);
            // ree_log(LOG_DEBUG, "%s c_ind_1 %d", __func__, c_ind_1);
            // ree_log(LOG_DEBUG, "%s c_ind_2 %d", __func__, c_ind_2);
            // ree_log(LOG_DEBUG, "%s c_ind_3 %d", __func__, c_ind_3);

            ree_cpy(p_mat_pack_b, p_mat_aux_b+n_ind_0, sizeof(uint32_t)*pack_widths);
            ree_cpy(p_mat_pack_b+pack_widths, p_mat_aux_b+n_ind_1, sizeof(uint32_t)*pack_widths);
            ree_cpy(p_mat_pack_b+pack_widths*2, p_mat_aux_b+n_ind_2, sizeof(uint32_t)*pack_widths);
            ree_cpy(p_mat_pack_b+pack_widths*3, p_mat_aux_b+n_ind_3, sizeof(uint32_t)*pack_widths);

            pack_vec_aux_b = vld4q_u32(p_mat_pack_b);
            pack_vec_aux_c.val[0] = vld1q_u32(p_mat_aux_c+c_ind_0);
            pack_vec_aux_c.val[1] = vld1q_u32(p_mat_aux_c+c_ind_1);
            pack_vec_aux_c.val[2] = vld1q_u32(p_mat_aux_c+c_ind_2);
            pack_vec_aux_c.val[3] = vld1q_u32(p_mat_aux_c+c_ind_3);
            for (int pack_ind = 0; pack_ind<pack_heights; pack_ind++)
            {
                if (pack_ind<remain_k_lane)
                {
                    pack_vec_aux_b.val[pack_ind] = vmulq_n_u32(pack_vec_aux_b.val[pack_ind], p_pack_params->alpha);
                }
                else
                {
                    pack_vec_aux_b.val[pack_ind] = vdupq_n_u32(0); 
                }
            }

            pack_vec_aux_c.val[0] = vmlaq_lane_u32(pack_vec_aux_c.val[0],
                                                   pack_vec_aux_b.val[0], 
                                                   vget_low_u32(pack_vec_aux_a.val[0]), 0);
            pack_vec_aux_c.val[0] = vmlaq_lane_u32(pack_vec_aux_c.val[0],
                                                   pack_vec_aux_b.val[1], 
                                                   vget_low_u32(pack_vec_aux_a.val[0]), 1);
            pack_vec_aux_c.val[0] = vmlaq_lane_u32(pack_vec_aux_c.val[0],
                                                   pack_vec_aux_b.val[2], 
                                                   vget_high_u32(pack_vec_aux_a.val[0]), 0);
            pack_vec_aux_c.val[0] = vmlaq_lane_u32(pack_vec_aux_c.val[0],
                                                   pack_vec_aux_b.val[3], 
                                                   vget_high_u32(pack_vec_aux_a.val[0]), 1);

            pack_vec_aux_c.val[1] = vmlaq_lane_u32(pack_vec_aux_c.val[1],
                                                   pack_vec_aux_b.val[0], 
                                                   vget_low_u32(pack_vec_aux_a.val[1]), 0);
            pack_vec_aux_c.val[1] = vmlaq_lane_u32(pack_vec_aux_c.val[1],
                                                   pack_vec_aux_b.val[1], 
                                                   vget_low_u32(pack_vec_aux_a.val[1]), 1);
            pack_vec_aux_c.val[1] = vmlaq_lane_u32(pack_vec_aux_c.val[1],
                                                   pack_vec_aux_b.val[2], 
                                                   vget_high_u32(pack_vec_aux_a.val[1]), 0);
            pack_vec_aux_c.val[1] = vmlaq_lane_u32(pack_vec_aux_c.val[1],
                                                   pack_vec_aux_b.val[3], 
                                                   vget_high_u32(pack_vec_aux_a.val[1]), 1);

            pack_vec_aux_c.val[2] = vmlaq_lane_u32(pack_vec_aux_c.val[2],
                                                   pack_vec_aux_b.val[0], 
                                                   vget_low_u32(pack_vec_aux_a.val[2]), 0);
            pack_vec_aux_c.val[2] = vmlaq_lane_u32(pack_vec_aux_c.val[2],
                                                   pack_vec_aux_b.val[1], 
                                                   vget_low_u32(pack_vec_aux_a.val[2]), 1);
            pack_vec_aux_c.val[2] = vmlaq_lane_u32(pack_vec_aux_c.val[2],
                                                   pack_vec_aux_b.val[2], 
                                                   vget_high_u32(pack_vec_aux_a.val[2]), 0);
            pack_vec_aux_c.val[2] = vmlaq_lane_u32(pack_vec_aux_c.val[2],
                                                   pack_vec_aux_b.val[3], 
                                                   vget_high_u32(pack_vec_aux_a.val[2]), 1);

            pack_vec_aux_c.val[3] = vmlaq_lane_u32(pack_vec_aux_c.val[3],
                                                   pack_vec_aux_b.val[0], 
                                                   vget_low_u32(pack_vec_aux_a.val[3]), 0);
            pack_vec_aux_c.val[3] = vmlaq_lane_u32(pack_vec_aux_c.val[3],
                                                   pack_vec_aux_b.val[1], 
                                                   vget_low_u32(pack_vec_aux_a.val[3]), 1);
            pack_vec_aux_c.val[3] = vmlaq_lane_u32(pack_vec_aux_c.val[3],
                                                   pack_vec_aux_b.val[2], 
                                                   vget_high_u32(pack_vec_aux_a.val[3]), 0);
            pack_vec_aux_c.val[3] = vmlaq_lane_u32(pack_vec_aux_c.val[3],
                                                   pack_vec_aux_b.val[3], 
                                                   vget_high_u32(pack_vec_aux_a.val[3]), 1);
            
            vst1q_u32(p_mat_aux_c+c_ind_0, pack_vec_aux_c.val[0]);
            vst1q_u32(p_mat_aux_c+c_ind_1, pack_vec_aux_c.val[1]);
            vst1q_u32(p_mat_aux_c+c_ind_2, pack_vec_aux_c.val[2]);
            vst1q_u32(p_mat_aux_c+c_ind_3, pack_vec_aux_c.val[3]);
        }
    }

    end = now_ns();
    ree_log(LOG_DEBUG, "%s executes times %f", __func__, end - start);

EXIT_GEMM_CPU_NEON_NT_UINT32:
    ree_free(p_mat_aux_a);
    ree_free(p_mat_aux_b);
    ree_free(p_mat_pack_b);
    ree_free(p_vec_c);
    ANDROID_GEMM_FUNC_EXIT_LOG;
}

static void gemm_cpu_neon_tt_uint32(gemm_pack_param_t *p_pack_params)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    ree_log(LOG_ERROR, "%s doesn't support yet", __func__);
    ANDROID_GEMM_FUNC_EXIT_LOG;
}

void gemm_cpu_neon_uint32(gemm_param_metadata_t *p_metadata)
{
    ANDROID_GEMM_FUNC_ENTRACE_LOG;
    BOOL is_a_tran = p_metadata->is_a_tran;
    BOOL is_b_tran = p_metadata->is_b_tran;
    int m_ind = 0, n_ind = 0, k_ind = 0;
    int m_dims = p_metadata->m_dims;
    int n_dims = p_metadata->n_dims;
    int k_dims = p_metadata->k_dims;
    int alpha = p_metadata->alpha;
    int beta = p_metadata->beta;
    int out_ele_num = m_dims*n_dims;
    gemm_pack_param_t pack_params = {0};

    if (!p_metadata)
    {
        ree_log(LOG_ERROR, "%s occurs error with invalid parameters", __func__);
        goto EXIT_GEMM_CPU_NEON_UINT32;
    }

    if ((!p_metadata->mat_a)||(!p_metadata->mat_b)||(!p_metadata->mat_c))
    {
        ree_log(LOG_ERROR, "%s occurs error with invalid parameters NULL MATRIX", __func__);
        goto EXIT_GEMM_CPU_NEON_UINT32;
    }

    if ((m_dims<=0)||(n_dims<=0)||(k_dims<=0))
    {
        ree_log(LOG_ERROR, "%s occurs error with invalid parameters 0 dimensions", __func__);
        goto EXIT_GEMM_CPU_NEON_UINT32;
    }
    
    ree_log(LOG_DEBUG, "%s out_ele_num %d", __func__, out_ele_num);
    pack_params.m_dims = m_dims;
    pack_params.n_dims = n_dims;
    pack_params.k_dims = k_dims;
    pack_params.alpha = alpha;
    pack_params.beta = beta;
    pack_params.pack_heights = 4;
    pack_params.pack_widths = 4;
    pack_params.mat_a = (uint8_t*)p_metadata->mat_a;
    pack_params.mat_b = (uint8_t*)p_metadata->mat_b;
    pack_params.mat_c = (uint8_t*)p_metadata->mat_c;

    gemm_cpu_accu_neon_uint32(&pack_params);

    if ((!is_a_tran)&&(!is_b_tran))
    {
        gemm_cpu_neon_nn_uint32(&pack_params);
    }
    else if ((!is_a_tran)&&(is_b_tran))
    {
        gemm_cpu_neon_nt_uint32_v2(&pack_params);
    }
    else if ((is_a_tran)&&(!is_b_tran))
    {
        gemm_cpu_neon_tn_uint32(&pack_params);
    }
    else if ((is_a_tran)&&(is_b_tran))
    {
        gemm_cpu_neon_tt_uint32(&pack_params);
    }

    if (pack_params.mat_c_aux)
    {
        ree_log(LOG_DEBUG, "%s out_ele_num %d", __func__, out_ele_num);
        ree_cpy(pack_params.mat_c,
                pack_params.mat_c_aux,
                sizeof(uint32_t)*out_ele_num);
    } else {
        ree_log(LOG_ERROR, "%s occurs algorithm error pack_params.mat_c_aux is NULL", __func__);
    }

EXIT_GEMM_CPU_NEON_UINT32:
    ree_free(pack_params.mat_c_aux);
    ANDROID_GEMM_FUNC_EXIT_LOG;
}
