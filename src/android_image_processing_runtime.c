#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "android_arm_util.h"
#include "android_image_processing_runtime.h"
#include "android_image_processing.h"
#include "android_image_processing_im2col.h"
#include "android_image_processing_im2row.h"
#include "android_image_processing_gemm.h"

void hist_eq_runtime_c(void)
{
    FUNC_ENTRANCE_LOG;
    uint8_t *p_raw_img = NULL;
    uint8_t *p_his_eq_img = NULL;
    int read_size = 0;
    size_t file_size;
    FILE *file = NULL;
    
    file = ree_fopen(HISTOGRAM_EQUALIZATION_RAW_IMG, "rb");

    if (!file)
    {
        ree_log(LOG_ERROR, "%s FILE %s doesn't exist", __func__, HISTOGRAM_EQUALIZATION_RAW_IMG);
        goto EXIT_HIST_EQ_RUNTIME_C;
    }

    fseek(file, 0L, SEEK_END);
    file_size = (size_t)ftell(file);
    ree_log(LOG_DEBUG, "%s FILE's size is %zu", __func__, file_size);
    ree_fclose(file);

    file = ree_fopen(HISTOGRAM_EQUALIZATION_RAW_IMG, "rb");
    
    p_raw_img = ree_malloc(file_size);
    if (!p_raw_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_raw_img buffer failed", __func__);
        goto EXIT_HIST_EQ_RUNTIME_C;
    }
   
    read_size = fread(p_raw_img, sizeof(char), file_size, file);
    ree_fclose(file);
    ree_log(LOG_DEBUG, "%s read_size is %d", __func__, read_size);

    if (read_size != (int)file_size)
    {
        ree_log(LOG_ERROR, "%s reads file %s failed", __func__, HISTOGRAM_EQUALIZATION_RAW_IMG);
        goto EXIT_HIST_EQ_RUNTIME_C;
    }

    c_hist_eq(p_raw_img, &p_his_eq_img, file_size);

    file = ree_fopen(HISTOGRAM_EQUALIZATION_RAW_IMG_C, "wb");
    if (file && p_his_eq_img) 
    {
        ree_log(LOG_DEBUG, "%s saves histogram equalization result %s", __func__, HISTOGRAM_EQUALIZATION_RAW_IMG_C);
        fwrite(p_his_eq_img, sizeof(uint8_t), file_size, file);
        ree_fclose(file);
    }
    else 
    {
        ree_log(LOG_ERROR, "%s prepares to save %s failed", __func__, HISTOGRAM_EQUALIZATION_RAW_IMG_C);
    }

EXIT_HIST_EQ_RUNTIME_C:
    ree_free(p_raw_img);
    ree_free(p_his_eq_img);
    FUNC_EXIT_LOG;
}

void hist_eq_runtime_neon(void)
{
    FUNC_ENTRANCE_LOG;
    uint8_t *p_raw_img = NULL;
    uint8_t *p_his_eq_img = NULL;
    int read_size = 0;
    size_t file_size;
    FILE *file = NULL;
    
    file = ree_fopen(HISTOGRAM_EQUALIZATION_RAW_IMG, "rb");

    if (!file)
    {
        ree_log(LOG_ERROR, "%s FILE %s doesn't exist", __func__, HISTOGRAM_EQUALIZATION_RAW_IMG);
        goto EXIT_HIST_EQ_RUNTIME_NEON;
    }

    fseek(file, 0L, SEEK_END);
    file_size = (size_t)ftell(file);
    ree_log(LOG_DEBUG, "%s FILE's size is %zu", __func__, file_size);
    ree_fclose(file);

    file = ree_fopen(HISTOGRAM_EQUALIZATION_RAW_IMG, "rb");
    
    p_raw_img = ree_malloc(file_size);
    if (!p_raw_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_raw_img buffer failed", __func__);
        goto EXIT_HIST_EQ_RUNTIME_NEON;
    }
   
    read_size = fread(p_raw_img, sizeof(char), file_size, file);
    ree_fclose(file);
    ree_log(LOG_DEBUG, "%s read_size is %d", __func__, read_size);

    if (read_size != (int)file_size)
    {
        ree_log(LOG_ERROR, "%s reads file %s failed", __func__, HISTOGRAM_EQUALIZATION_RAW_IMG);
        goto EXIT_HIST_EQ_RUNTIME_NEON;
    }

    neon_hist_eq(p_raw_img, &p_his_eq_img, file_size);

    file = ree_fopen(HISTOGRAM_EQUALIZATION_RAW_IMG_NEON, "wb");
    if (file && p_his_eq_img) 
    {
        ree_log(LOG_DEBUG, "%s saves histogram equalization result %s", __func__, HISTOGRAM_EQUALIZATION_RAW_IMG_NEON);
        fwrite(p_his_eq_img, sizeof(uint8_t), file_size, file);
        ree_fclose(file);
    }
    else 
    {
        ree_log(LOG_ERROR, "%s prepares to save %s failed", __func__, HISTOGRAM_EQUALIZATION_RAW_IMG_NEON);
    }

EXIT_HIST_EQ_RUNTIME_NEON:
    ree_free(p_raw_img);
    ree_free(p_his_eq_img);
    FUNC_EXIT_LOG;
}

void hist_matching_runtime_c(void)
{
    FUNC_ENTRANCE_LOG;
    size_t in_file_size = 0;
    size_t ref_file_size = 0;
    size_t in_read_size = 0;
    size_t ref_read_size = 0;
    uint8_t *p_in_raw_img = NULL;
    uint8_t *p_ref_raw_img = NULL;
    uint8_t *p_in_match_res_img = NULL;
    FILE *in_file = NULL;
    FILE *ref_file = NULL;
    FILE *res_file = NULL;

    ree_check_fopen(in_file, 
                    HISTOGRAM_MATCHING_IN_RAW_IMG, 
                    "rb", 
                    EXIT_HIST_MATCHING_RUNTIME_C);
    ree_file_size(in_file_size, in_file);
    
    p_in_raw_img = ree_malloc((int)in_file_size);
    if (!p_in_raw_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_in_raw_img buffer failed", __func__);
        goto EXIT_HIST_MATCHING_RUNTIME_C;
    }
    ree_set(p_in_raw_img, 0, (int)in_file_size);

    ree_check_fopen(in_file, 
                    HISTOGRAM_MATCHING_IN_RAW_IMG, 
                    "rb", 
                    EXIT_HIST_MATCHING_RUNTIME_C);
    ree_file_read(in_file, p_in_raw_img, (int)in_file_size, in_read_size);

    ree_check_fopen(ref_file, 
                    HISTOGRAM_MATCHING_REF_RAW_IMG, 
                    "rb", 
                    EXIT_HIST_MATCHING_RUNTIME_C);
    ree_file_size(ref_file_size, ref_file);
    
    p_ref_raw_img = ree_malloc((int)ref_file_size);
    if (!p_ref_raw_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_ref_raw_img buffer failed", __func__);
        goto EXIT_HIST_MATCHING_RUNTIME_C;
    }
    ree_set(p_ref_raw_img, 0, (int)ref_file_size);

    ree_check_fopen(ref_file, 
                    HISTOGRAM_MATCHING_REF_RAW_IMG, 
                    "rb", 
                    EXIT_HIST_MATCHING_RUNTIME_C);
    ree_file_read(ref_file, p_ref_raw_img, (int)ref_file_size, ref_read_size);
    c_hist_match(p_in_raw_img,
                 p_ref_raw_img,
                 &p_in_match_res_img,
                 in_read_size,
                 ref_read_size);

    ree_check_fopen(res_file, 
                    HISTOGRAM_MATCHING_RES_RAW_IMG_C, 
                    "wb", 
                    EXIT_HIST_MATCHING_RUNTIME_C);
    ree_file_write(res_file,
                   p_in_match_res_img,
                   HISTOGRAM_MATCHING_RES_RAW_IMG_C,
                   in_read_size);


EXIT_HIST_MATCHING_RUNTIME_C:
    ree_free(p_in_raw_img);
    ree_free(p_ref_raw_img);
    ree_free(p_in_match_res_img);
    FUNC_EXIT_LOG;
}

void hist_matching_runtime_neon()
{
    FUNC_ENTRANCE_LOG;
    size_t in_file_size = 0;
    size_t ref_file_size = 0;
    size_t in_read_size = 0;
    size_t ref_read_size = 0;
    uint8_t *p_in_raw_img = NULL;
    uint8_t *p_ref_raw_img = NULL;
    uint8_t *p_in_match_res_img = NULL;
    FILE *in_file = NULL;
    FILE *ref_file = NULL;
    FILE *res_file = NULL;

    ree_check_fopen(in_file, 
                    HISTOGRAM_MATCHING_IN_RAW_IMG, 
                    "rb", 
                    EXIT_HIST_MATCHING_RUNTIME_NEON);
    ree_file_size(in_file_size, in_file);
    
    p_in_raw_img = ree_malloc((int)in_file_size);
    if (!p_in_raw_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_in_raw_img buffer failed", __func__);
        goto EXIT_HIST_MATCHING_RUNTIME_NEON;
    }
    ree_set(p_in_raw_img, 0, (int)in_file_size);

    ree_check_fopen(in_file, 
                    HISTOGRAM_MATCHING_IN_RAW_IMG, 
                    "rb", 
                    EXIT_HIST_MATCHING_RUNTIME_NEON);
    ree_file_read(in_file, p_in_raw_img, (int)in_file_size, in_read_size);

    ree_check_fopen(ref_file, 
                    HISTOGRAM_MATCHING_REF_RAW_IMG, 
                    "rb", 
                    EXIT_HIST_MATCHING_RUNTIME_NEON);
    ree_file_size(ref_file_size, ref_file);
    
    p_ref_raw_img = ree_malloc((int)ref_file_size);
    if (!p_ref_raw_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_ref_raw_img buffer failed", __func__);
        goto EXIT_HIST_MATCHING_RUNTIME_NEON;
    }
    ree_set(p_ref_raw_img, 0, (int)ref_file_size);

    ree_check_fopen(ref_file, 
                    HISTOGRAM_MATCHING_REF_RAW_IMG, 
                    "rb", 
                    EXIT_HIST_MATCHING_RUNTIME_NEON);
    ree_file_read(ref_file, p_ref_raw_img, (int)ref_file_size, ref_read_size);


    neon_hist_match(p_in_raw_img,
                    p_ref_raw_img,
                    &p_in_match_res_img,
                    in_read_size,
                    ref_read_size);

    // ree_check_fopen(res_file, 
    //                 HISTOGRAM_MATCHING_RES_RAW_IMG_NEON, 
    //                 "wb", 
    //                 EXIT_HIST_MATCHING_RUNTIME_NEON);
    // ree_file_write(res_file,
    //                p_in_match_res_img,
    //                HISTOGRAM_MATCHING_RES_RAW_IMG_NEON,
    //                in_read_size);


EXIT_HIST_MATCHING_RUNTIME_NEON:
    ree_free(p_in_raw_img);
    ree_free(p_ref_raw_img);
    ree_free(p_in_match_res_img);
    FUNC_EXIT_LOG;
}

void im2col_cpu_runtime(void)
{
    FUNC_ENTRANCE_LOG;
    uint8_t *img_data = NULL;
    uint8_t *col_data = NULL;

    int channels = 1, height = 3, width = 3;
    int kernel_size = 2, stride = 1, padding = 1;
    int out_height = 0, out_width = 0;
    int input_num_element = 0;
    int output_num_element = 0;
    int workplace_size = 0;
    int inputs_size = 0;

    input_num_element = height * width * channels;
    inputs_size = input_num_element * sizeof(uint8_t);
    ree_log(LOG_DEBUG, "%s input_num_element %d", __func__, input_num_element);
    ree_log(LOG_DEBUG, "%s input_size %d", __func__, inputs_size);
    
    out_height = conv_2d_out_height(height,
                                    padding, 
                                    kernel_size,
                                    stride);
    out_width = conv_2d_out_width(width,
                                  padding,
                                  kernel_size,
                                  stride);
    ree_log(LOG_DEBUG, "%s out_height %d out_width %d", __func__,
                                                        out_height,
                                                        out_width);
    output_num_element = out_height * out_width * kernel_size * kernel_size * channels;
    workplace_size = output_num_element * sizeof(uint8_t);
    ree_log(LOG_DEBUG, "%s worplace_size %d", __func__, workplace_size);

    img_data = (uint8_t*)ree_malloc(inputs_size);
    ree_set(img_data, 0, inputs_size);
    if (!img_data)
    {
        ree_log(LOG_ERROR, "%s allocates img_data failed", __func__);
        goto EXIT_IM2COL_CPU_RUNTIME;
    }

    col_data = (uint8_t*)ree_malloc(workplace_size);
    ree_set(col_data, 0, workplace_size);
    if (!col_data)
    {
        ree_log(LOG_ERROR, "%s allocates col_data failed", __func__);
        goto EXIT_IM2COL_CPU_RUNTIME;
    }

    for (int i = 0; i < input_num_element; i++)
    {
        img_data[i] = i;
        ree_printf(LOG_DEBUG, "%d ", img_data[i]);
    }
    ree_printf(LOG_DEBUG, "\n");

    im2col_cpu_uint8_t(img_data,
                       channels,
                       height,
                       width,
                       kernel_size,
                       stride,
                       padding, 
                       col_data);

    for (int i = 0; i < output_num_element; i++)
    {
        ree_printf(LOG_DEBUG, "%d ", col_data[i]);
    }
    ree_printf(LOG_DEBUG, "\n");


EXIT_IM2COL_CPU_RUNTIME:
    ree_free(img_data);
    ree_free(col_data);

    FUNC_EXIT_LOG;
}

void im2row_cpu_runtime(void)
{
    FUNC_ENTRANCE_LOG;
    
    uint8_t *p_input_img = NULL;
    uint8_t *p_output_row = NULL;
    uint8_t *p_output_img = NULL;
    unsigned int input_highs = 3, input_widths = 3, input_channels = 1;
    unsigned int input_elem_num = input_highs * input_widths * input_channels;
    unsigned int kernel_size = 2;
    unsigned int stride = 1, padding = 1;

    p_input_img = ree_malloc(sizeof(uint8_t)*input_elem_num);

    if (!p_input_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_input_img failed", __func__);
        goto EXIT_IM2ROW_CPU_RUNTIME;
    }

    for (int i = 0; i<input_elem_num; i++)
        p_input_img[i] = i;

    im2row_cpu_c(p_input_img,
                 input_highs, input_widths, input_channels,
                 kernel_size, stride, padding,
                 &p_output_row);
    row2im_cpu_c(p_output_row,
                 input_highs, input_widths, input_channels,
                 kernel_size, stride, padding,
                 &p_output_img);
    if (p_output_img)
    {
        for (int i = 0; i<input_elem_num; i++)
        {
            printf("%d ", p_output_img[i]);
        }
    }

EXIT_IM2ROW_CPU_RUNTIME:
    ree_free(p_input_img);
    ree_free(p_output_row);
    ree_free(p_output_img);
    FUNC_EXIT_LOG;
}

void gemm_cpu_c_uint32_runtime(void)
{
    FUNC_ENTRANCE_LOG;
    // N-N
    //  uint32_t mat_a[9] = { 3, 4, 5, 
    //                        1,10, 6,
    //                       12, 9, 7};
    //  uint32_t mat_b[9] = { 2, 4,12,
    //                       21,45,21,
    //                       32,12, 4};
    // N-T
    // uint32_t mat_a[9] = { 3, 4, 5, 
    //                       1,10, 6,
    //                      12, 9, 7};
    // uint32_t mat_b[9] = { 2,21,32,
    //                       4,45,12,
    //                      12,21, 4};
    // T-N
    // uint32_t mat_a[9] = { 3, 1,12, 
    //                       4,10, 9,
    //                       5, 6, 7};
    // uint32_t mat_b[9] = { 2, 4,12,
    //                      21,45,21,
    //                      32,12, 4};
    // T-T
    uint32_t mat_a[9] = { 3, 1,12, 
                          4,10, 9,
                          5, 6, 7};
    uint32_t mat_b[9] = { 2,21,32,
                          4,45,12,
                         12,21, 4};

    uint32_t mat_c[9] = {0};
    
    gemm_param_metadata_t gemm_param = {0};
    gemm_param.is_a_tran = TRUE;
    gemm_param.is_b_tran = TRUE;
    gemm_param.m_dims = 3;
    gemm_param.n_dims = 3;
    gemm_param.k_dims = 3;
    gemm_param.alpha = 1;
    gemm_param.beta = 1;
    gemm_param.mat_a = (uint8_t*)&mat_a;
    gemm_param.mat_b = (uint8_t*)&mat_b;
    gemm_param.mat_c = (uint8_t*)&mat_c;

    gemm_cpu_c_uint32(&gemm_param);

    for (int i = 0; i<gemm_param.m_dims; i++)
    {
        for (int j = 0; j<gemm_param.n_dims; j++)
        {
            printf("%d ", mat_c[i*gemm_param.n_dims+j]);
        }
        printf("\n");
    }

EXIT_GEMM_CPU_C_UINT32_RUNTIME:
    FUNC_EXIT_LOG;
}

void gemm_cpu_neon_uint32_runtime(void)
{
    FUNC_ENTRANCE_LOG;
    // N-T
    uint32_t mat_a[9] = { 3, 4, 5,
                          1,10, 6,
                         12, 9, 7};
    uint32_t mat_b[9] = { 2,21,32,
                          4,45,12,
                         12,21, 4};
    uint32_t mat_c[9] = {0};
    gemm_param_metadata_t gemm_param = {0};
    gemm_param.is_a_tran = FALSE;
    gemm_param.is_b_tran = TRUE;
    gemm_param.m_dims = 3;
    gemm_param.n_dims = 3;
    gemm_param.k_dims = 3;
    gemm_param.alpha = 1;
    gemm_param.beta = 0;
    gemm_param.mat_a = (uint8_t*)&mat_a;
    gemm_param.mat_b = (uint8_t*)&mat_b;
    gemm_param.mat_c = (uint8_t*)&mat_c;

    gemm_cpu_neon_uint32(&gemm_param);

    for (int i = 0; i<gemm_param.m_dims; i++)
    {
        for (int j = 0; j<gemm_param.n_dims; j++)
        {
            printf("%d ", mat_c[i*gemm_param.n_dims+j]);
        }
        printf("\n");
    }

EXIT_GEMM_CPU_NEON_UINT32_RUNTIME:
    FUNC_EXIT_LOG;
}

void gemm_im2row_cpu_c_uint32_runtime(void)
{
    FUNC_ENTRANCE_LOG;
    uint32_t *p_input_img = NULL;
    uint32_t *p_input_img_row = NULL;
    uint32_t *p_input_filters = NULL;
    uint32_t *p_output_row_features = NULL;
    unsigned int input_heights = 7, input_widths = 7, input_channels = 1;
    unsigned int input_ele_num = input_heights*input_widths*input_channels;
    unsigned int kernel_size = 3, kernel_num = 3;
    unsigned int output_heights = 0, output_widths = 0;
    unsigned int padding = 1, stride = 1;
    size_t in_file_size = 0, in_r_file_size = 0;
    size_t out_file_size = 0, out_w_file_size = 0;
    FILE *input_img_file = NULL;
    FILE *input_kernels_file = NULL;
    FILE *output_features_file = NULL;
    gemm_param_metadata_t gemm_param = {0};

    output_heights = (input_heights+2*padding-kernel_size)/stride + 1;
    output_widths = (input_widths+2*padding-kernel_size)/stride + 1;
    ree_log(LOG_DEBUG, "%s output_heights %d output_widths %d", __func__,
                                                                output_heights,
                                                                output_widths);

    ree_check_fopen(input_img_file,
                    GEMM_CPU_TEST_INPUT_BIN,
                    "rb",
                    EXIT_GEMM_CPU_C_RUNTIME);
    ree_file_size(in_file_size,
                  input_img_file);
    
    if (in_file_size != (sizeof(uint32_t)*input_channels*input_widths*input_heights))
    {
        ree_log(LOG_ERROR, "%s occurs error when readding %s", __func__, GEMM_CPU_TEST_INPUT_BIN);
        goto EXIT_GEMM_CPU_C_RUNTIME;
    }
    ree_fclose(input_img_file);

    p_input_img = ree_malloc((int)in_file_size);
    if (!p_input_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_input_img buffer failed", __func__);
        goto EXIT_GEMM_CPU_C_RUNTIME;
    }

    ree_check_fopen(input_img_file,
                    GEMM_CPU_TEST_INPUT_BIN,
                    "rb",
                    EXIT_GEMM_CPU_C_RUNTIME);
    ree_file_read(input_img_file,
                  p_input_img,
                  (int)in_file_size,
                  in_r_file_size);
    ree_fclose(input_img_file);

    ree_check_fopen(input_kernels_file,
                    GEMM_CPU_TEST_FILTERS_BIN,
                    "rb",
                    EXIT_GEMM_CPU_C_RUNTIME);
    ree_file_size(in_file_size,
                  input_kernels_file);

    if (in_file_size != (sizeof(uint32_t)*kernel_size*kernel_size*kernel_num))
    {
        ree_log(LOG_ERROR, "%s occurs error when readding %s", __func__, GEMM_CPU_TEST_FILTERS_BIN);
        goto EXIT_GEMM_CPU_C_RUNTIME;
    }
    ree_fclose(input_kernels_file);

    p_input_filters = ree_malloc((int)in_file_size);
    if (!p_input_filters)
    {
        ree_log(LOG_ERROR, "%s allocates p_input_filters buffer failed", __func__);
        goto EXIT_GEMM_CPU_C_RUNTIME;
    }


    ree_check_fopen(input_kernels_file,
                    GEMM_CPU_TEST_FILTERS_BIN,
                    "rb",
                    EXIT_GEMM_CPU_C_RUNTIME);
    ree_file_read(input_kernels_file,
                  p_input_filters,
                  (int)in_file_size,
                  in_r_file_size);
    ree_fclose(input_kernels_file);

    im2row_cpu_c_u32(p_input_img,
                     input_heights, input_widths, input_channels,
                     kernel_size, stride, padding,
                     &p_input_img_row);

    p_output_row_features = ree_malloc(sizeof(uint32_t)*output_heights*output_widths*kernel_num);

    if (!p_output_row_features)
    {
        ree_log(LOG_DEBUG, "%s allocates p_output_row_features failed", __func__);
        goto EXIT_GEMM_CPU_C_RUNTIME; 
    }
    ree_set(p_output_row_features, 0, sizeof(uint32_t)*output_heights*output_widths*kernel_num);

    gemm_param.is_a_tran = FALSE;
    gemm_param.is_b_tran = TRUE;
    gemm_param.m_dims = kernel_num;
    gemm_param.n_dims = output_heights*output_widths;
    gemm_param.k_dims = kernel_size*kernel_size;
    gemm_param.alpha = 1;
    gemm_param.beta = 0;
    gemm_param.mat_a = (uint8_t*)p_input_filters;
    gemm_param.mat_b = (uint8_t*)p_input_img_row;
    gemm_param.mat_c = (uint8_t*)p_output_row_features;

    gemm_cpu_c_uint32(&gemm_param);

    for (int m_ind = 0; m_ind<gemm_param.m_dims; m_ind++)
    {
        for (int n_ind = 0; n_ind<gemm_param.n_dims; n_ind++)
        {
            printf("%4d ", *((uint32_t*)p_output_row_features+m_ind*gemm_param.n_dims+n_ind));
        }
        printf("\n");
    }

EXIT_GEMM_CPU_C_RUNTIME:
    ree_free(p_input_img);
    ree_free(p_input_img_row);
    ree_free(p_input_filters);
    ree_free(p_output_row_features);
    FUNC_EXIT_LOG;
}


void gemm_im2row_cpu_neon_uint32_runtime(void)
{
    FUNC_ENTRANCE_LOG;
    uint32_t *p_input_img = NULL;
    uint32_t *p_input_img_row = NULL;
    uint32_t *p_input_filters = NULL;
    uint32_t *p_output_row_features = NULL;
    unsigned int input_heights = 7, input_widths = 7, input_channels = 1;
    unsigned int input_ele_num = input_heights*input_widths*input_channels;
    unsigned int kernel_size = 3, kernel_num = 3;
    unsigned int output_heights = 0, output_widths = 0;
    unsigned int padding = 1, stride = 1;
    size_t in_file_size = 0, in_r_file_size = 0;
    size_t out_file_size = 0, out_w_file_size = 0;
    FILE *input_img_file = NULL;
    FILE *input_kernels_file = NULL;
    FILE *output_features_file = NULL;
    gemm_param_metadata_t gemm_param = {0};

    output_heights = (input_heights+2*padding-kernel_size)/stride + 1;
    output_widths = (input_widths+2*padding-kernel_size)/stride + 1;
    ree_log(LOG_DEBUG, "%s output_heights %d output_widths %d", __func__,
                                                                output_heights,
                                                                output_widths);

    ree_check_fopen(input_img_file,
                    GEMM_CPU_TEST_INPUT_BIN,
                    "rb",
                    EXIT_GEMM_CPU_NEON_RUNTIME);
    ree_file_size(in_file_size,
                  input_img_file);
    
    if (in_file_size != (sizeof(uint32_t)*input_channels*input_widths*input_heights))
    {
        ree_log(LOG_ERROR, "%s occurs error when readding %s", __func__, GEMM_CPU_TEST_INPUT_BIN);
        goto EXIT_GEMM_CPU_NEON_RUNTIME;
    }
    ree_fclose(input_img_file);

    p_input_img = ree_malloc((int)in_file_size);
    if (!p_input_img)
    {
        ree_log(LOG_ERROR, "%s allocates p_input_img buffer failed", __func__);
        goto EXIT_GEMM_CPU_NEON_RUNTIME;
    }

    ree_check_fopen(input_img_file,
                    GEMM_CPU_TEST_INPUT_BIN,
                    "rb",
                    EXIT_GEMM_CPU_NEON_RUNTIME);
    ree_file_read(input_img_file,
                  p_input_img,
                  (int)in_file_size,
                  in_r_file_size);
    ree_fclose(input_img_file);

    ree_check_fopen(input_kernels_file,
                    GEMM_CPU_TEST_FILTERS_BIN,
                    "rb",
                    EXIT_GEMM_CPU_NEON_RUNTIME);
    ree_file_size(in_file_size,
                  input_kernels_file);

    if (in_file_size != (sizeof(uint32_t)*kernel_size*kernel_size*kernel_num))
    {
        ree_log(LOG_ERROR, "%s occurs error when readding %s", __func__, GEMM_CPU_TEST_FILTERS_BIN);
        goto EXIT_GEMM_CPU_NEON_RUNTIME;
    }
    ree_fclose(input_kernels_file);

    p_input_filters = ree_malloc((int)in_file_size);
    if (!p_input_filters)
    {
        ree_log(LOG_ERROR, "%s allocates p_input_filters buffer failed", __func__);
        goto EXIT_GEMM_CPU_NEON_RUNTIME;
    }


    ree_check_fopen(input_kernels_file,
                    GEMM_CPU_TEST_FILTERS_BIN,
                    "rb",
                    EXIT_GEMM_CPU_NEON_RUNTIME);
    ree_file_read(input_kernels_file,
                  p_input_filters,
                  (int)in_file_size,
                  in_r_file_size);
    ree_fclose(input_kernels_file);

    im2row_cpu_c_u32(p_input_img,
                     input_heights, input_widths, input_channels,
                     kernel_size, stride, padding,
                     &p_input_img_row);

    p_output_row_features = ree_malloc(sizeof(uint32_t)*output_heights*output_widths*kernel_num);

    if (!p_output_row_features)
    {
        ree_log(LOG_DEBUG, "%s allocates p_output_row_features failed", __func__);
        goto EXIT_GEMM_CPU_NEON_RUNTIME; 
    }
    ree_set(p_output_row_features, 0, sizeof(uint32_t)*output_heights*output_widths*kernel_num);

    gemm_param.is_a_tran = FALSE;
    gemm_param.is_b_tran = TRUE;
    gemm_param.m_dims = kernel_num;
    gemm_param.n_dims = output_heights*output_widths;
    gemm_param.k_dims = kernel_size*kernel_size;
    gemm_param.alpha = 1;
    gemm_param.beta = 0;
    gemm_param.mat_a = (uint8_t*)p_input_filters;
    gemm_param.mat_b = (uint8_t*)p_input_img_row;
    gemm_param.mat_c = (uint8_t*)p_output_row_features;

    gemm_cpu_neon_uint32(&gemm_param);

    for (int m_ind = 0; m_ind<gemm_param.m_dims; m_ind++)
    {
        for (int n_ind = 0; n_ind<gemm_param.n_dims; n_ind++)
        {
            printf("%4d ", *((uint32_t*)p_output_row_features+m_ind*gemm_param.n_dims+n_ind));
        }
        printf("\n");
    }

EXIT_GEMM_CPU_NEON_RUNTIME:
    ree_free(p_input_img);
    ree_free(p_input_img_row);
    ree_free(p_input_filters);
    ree_free(p_output_row_features);
    FUNC_EXIT_LOG;
}
