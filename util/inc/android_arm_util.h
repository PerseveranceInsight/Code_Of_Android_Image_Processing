#include <android/log.h>
#ifndef __ANDROID_ARM_UTIL_H__
#define __ANDROID_ARM_UTIL_H__
#ifdef __DEBUG__
#define LOG_ERROR       ANDROID_LOG_ERROR
#define LOG_INFO        ANDROID_LOG_INFO
#define LOG_DEBUG       ANDROID_LOG_DEBUG
#define LOG_VERBOSE     ANDOIRD_LOG_VERBOSE
#define ree_log(LOG_LEVEL, x...)                                    \
    {                                                               \
        __android_log_print(LOG_LEVEL, "ANDROID_IPP", x);           \
        printf(x);                                                  \
        printf("\n");                                               \
    }

#define FUNC_ENTRANCE_LOG       ree_log(LOG_DEBUG, "%s enters", __func__);
#define FUNC_EXIT_LOG           ree_log(LOG_DEBUG, "%s leaves", __func__);
#define ree_printf(LOG_LEVEL, x...)                                 \
{                                                                   \
    printf(x);                                                      \
}
#else
#define ree_log(LOG_LEVEL, x...) do {} while (0)
#endif
#ifndef NULL
#define NULL                                                ((void*)0)
#endif
#define ree_malloc(size)                                    malloc(size)
#define ree_set(dst, value, size)                           memset(dst, value, size)
#define ree_cpy(dst, src, size)                             memcpy(dst, src, size)
#define ree_free(src)                                                               \
{                                                                                   \
    if (src)                                                                        \
    {                                                                               \
        free(src);                                                                  \
        src = NULL;                                                                 \
    }                                                                               \
}

#define ree_fopen(file_name, mode)                         fopen(file_name, mode)   
#define ree_fclose(src_file)                                                        \
{                                                                                   \
    if (src_file)                                                                   \
    {                                                                               \
        fclose(src_file);                                                           \
        src_file = NULL;                                                            \
    }                                                                               \
}

#define ree_check_fopen(file_ptr, file_name, mode, EXIT_TAG)                        \
{                                                                                   \
    file_ptr = ree_fopen(file_name, mode);                                          \
    if (!file_ptr)                                                                  \
    {                                                                               \
        ree_log(LOG_ERROR, "%s can't open %s", __func__, file_name);                \
        goto EXIT_TAG;                                                              \
    }                                                                               \
}

#define ree_file_size(file_size, file_ptr)                                          \
{                                                                                   \
    fseek(file_ptr, 0L, SEEK_END);                                                  \
    file_size = (size_t)ftell(file_ptr);                                            \
    ree_log(LOG_DEBUG, "%s FILE's size is %zu", __func__, file_size);               \
    ree_fclose(file_ptr);                                                           \
}

#define ree_file_read(file_ptr, p_buffer, file_size, read_size)                     \
{                                                                                   \
    read_size = fread(p_buffer, sizeof(char), file_size, file_ptr);                 \
    ree_fclose(file_ptr);                                                           \
    ree_log(LOG_DEBUG, "%s read_size is %zu", __func__, read_size);                 \
}

#define ree_file_write(file_ptr, p_buffer, file_name, file_size)                    \
{                                                                                   \
    if (file_ptr && p_buffer)                                                       \
    {                                                                               \
        ree_log(LOG_DEBUG, "%s saves %s", __func__, file_name);                     \
        fwrite(p_buffer, sizeof(uint8_t), file_size, file_ptr);                     \
        ree_fclose(file_ptr);                                                       \
    } else                                                                          \
    {                                                                               \
        ree_log(LOG_ERROR, "%s prepares to save %s failed", __func__, file_name);   \
    }                                                                               \
}

double now_ns(void);
#endif
