#ifndef __ANDROID_IMAGE_PROCESSING_GEMM_H__
#define __ANDROID_IMAGE_PROCESSING_GEMM_H__
#include "android_arm_util.h"
#include "android_arm_typedef.h"

#define ANDROID_GEMM_LOG_LEVEL LOG_DEBUG

#define ANDROID_GEMM_FUNC_ENTRACE_LOG FUNC_ENTRANCE_LOG
#define ANDROID_GEMM_FUNC_EXIT_LOG FUNC_EXIT_LOG

typedef struct gemm_param_submetadata {
    int m_dims;
    int n_dims;
    int k_dims;
    int alpha;
    uint8_t *mat_a;
    uint8_t *mat_b;
    uint8_t *mat_c;
} gemm_param_submetadata_t;

typedef struct gemm_pack_param {
    int m_dims;
    int n_dims;
    int k_dims;
    int alpha;
    int beta;
    int pack_heights;
    int pack_widths;
    uint8_t *mat_a;
    uint8_t *mat_b;
    uint8_t *mat_c;
    uint8_t *mat_c_aux;
} gemm_pack_param_t;

typedef struct gemm_param_metadata {
    BOOL is_a_tran;
    BOOL is_b_tran;
    int m_dims;
    int n_dims;
    int k_dims;
    int alpha;
    int beta;
    uint8_t *mat_a;
    uint8_t *mat_b;
    uint8_t *mat_c;
} gemm_param_metadata_t;

void gemm_cpu_c_uint32(gemm_param_metadata_t*);
void gemm_cpu_neon_uint32(gemm_param_metadata_t*);
void gemm_cpu_neon_uint32_v2(gemm_param_metadata_t*);
#endif
