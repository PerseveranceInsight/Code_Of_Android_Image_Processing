#ifndef __ANDROID_IMAGE_PROCESSING_PACK_H__
#define __ANDROID_IMAGE_PROCESSING_PACK_H__
#include <stdint.h>
#include "android_arm_util.h"

#ifdef EN_PACK_DEBUG
#define PACK_FUNC_ENTRANCE_LOG FUNC_ENTRANCE_LOG
#define PACK_FUNC_EXIT_LOG FUNC_EXIT_LOG
#else
#define PACK_FUNC_ENTRANCE_LOG do {} while(0)
#define PACK_FUNC_EXIT_LOG do {} while(0)
#endif

void pack_cpu_c_uint32_t(uint32_t const *in_unpack_mat, 
                         const int in_heights,
                         const int in_widths,
                         const int pack_heights,
                         const int pack_widths,
                         uint32_t *out_pack_mat);

void pack_cpu_neon_uint32_t(uint32_t const *in_unpack_mat,
                            const int in_heights,
                            const int in_widths,
                            const int pack_heights,
                            const int pack_widths,
                            uint32_t **out_pack_mat);

void unpack_cpu_c_uint32_t(uint32_t const *in_pack_mat,
                           const int in_heights,
                           const int in_widths,
                           const int pack_heights,
                           const int pack_widths,
                           uint32_t **out_unpack_mat);

void unpack_cpu_neon_uint32_t(uint32_t const *in_pack_mat,
                              const int in_heights,
                              const int in_widths,
                              const int pack_heights,
                              const int pack_widths,
                              uint32_t **out_unpack_mat);


#endif
