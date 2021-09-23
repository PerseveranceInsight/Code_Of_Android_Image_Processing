#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "android_arm_util.h"
#include "android_image_processing_runtime.h"

int main(int argc, char* argv[])
{
    FUNC_ENTRANCE_LOG;
#ifdef EXEC_HIST_EQ
    hist_eq_runtime_c();
    hist_eq_runtime_neon();
#elif EXEC_HIST_MATCH
    hist_matching_runtime_c();
    hist_matching_runtime_neon();
#elif EXEC_IM2COL
    im2col_cpu_runtime();
#elif EXEC_IM2ROW
    im2row_cpu_runtime();
#elif EXEC_GEMM_CPU
    gemm_cpu_c_uint32_runtime();
    gemm_cpu_neon_uint32_runtime();
#endif
    gemm_im2row_cpu_c_uint32_runtime();
    gemm_im2row_cpu_neon_uint32_runtime();
EXIT_MAIN:
    FUNC_EXIT_LOG;
    return 0;
}
