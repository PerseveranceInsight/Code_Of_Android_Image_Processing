#ifndef __ANDROID_IMAGE_PROCESSING_RUNTIME_H__
#define __ANDROID_IMAGE_PROCESSING_RUNTIME_H__

#define HISTOGRAM_EQUALIZATION_RAW_IMG                  "/vendor/bin/arm_ipp/ipp_histogram_sample.bin"
#define HISTOGRAM_EQUALIZATION_RAW_IMG_C                "/vendor/bin/arm_ipp/ipp_his_eq_res_c.bin"
#define HISTOGRAM_EQUALIZATION_RAW_IMG_NEON             "/vendor/bin/arm_ipp/ipp_his_eq_res_neon.bin"

#define HISTOGRAM_MATCHING_IN_RAW_IMG                   "/vendor/bin/arm_ipp/ipp_histogram_matching_in.bin"
#define HISTOGRAM_MATCHING_REF_RAW_IMG                  "/vendor/bin/arm_ipp/ipp_histogram_matching_ref.bin"
#define HISTOGRAM_MATCHING_RES_RAW_IMG_C                "/vendor/bin/arm_ipp/ipp_his_mat_res_c.bin"
#define HISTOGRAM_MATCHING_RES_RAW_IMG_NEON             "/vendor/bin/arm_ipp/ipp_his_mat_res_neon.bin"

#define GEMM_CPU_TEST_INPUT_BIN                         "/vendor/bin/arm_ipp/ipp_gemm_test_input.bin"
#define GEMM_CPU_TEST_FILTERS_BIN                       "/vendor/bin/arm_ipp/ipp_gemm_test_filters.bin"
#define GEMM_CPU_TEST_C_OUTPUT_BIN                      "/vendor/bin/arm_ipp/ipp_gemm_test_outputs_c.bin"
#define GEMM_CPU_TEST_NEON_OUTPUT_BIN                   "/vendor/bin/arm_ipp/ipp_gemm_test_outputs_neon.bin"

void hist_eq_runtime_c(void);
void hist_eq_runtime_neon(void);
void hist_matching_runtime_c(void);
void hist_matching_runtime_neon(void);

void im2col_cpu_runtime(void);
void im2row_cpu_runtime(void);
void gemm_cpu_c_uint32_runtime(void);
void gemm_cpu_neon_uint32_runtime(void);
void gemm_im2row_cpu_c_uint32_runtime(void);
void gemm_im2row_cpu_neon_uint32_runtime(void);
#endif
