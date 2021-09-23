LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := android_ipp

LOCAL_CFLAGS += -D__DEBUG__
LOCAL_CFLAGS += -Wall -Wno-unused-function -Wno-unused-variable -Wno-unused-label -Wno-return-type
# LOCAL_CFLAGS += -Wsign-compare -Wunused-function -Wunused-label -Wimplicit-fallthrough -Wstrict-prototypes
ifeq ($(EN_DEBUG_SYM), true)
LOCAL_CFLAGS += -g -ggdb
endif

LOCAL_ARM_MODE := arm

PROJECT_SRC = $(LOCAL_PATH)/../src
PROJECT_INC = $(LOCAL_PATH)/../inc
PROJECT_UTIL_INC = $(LOCAL_PATH)/../util/inc
PROJECT_UTIL_SRC = $(LOCAL_PATH)/../util/src

LOCAL_C_INCLUDES += $(PROJECT_INC) \
					$(PROJECT_UTIL_INC) \

LOCAL_SRC_FILES += $(PROJECT_SRC)/android_image_processing_main.c \
				   $(PROJECT_SRC)/android_image_processing_runtime.c \
				   $(PROJECT_SRC)/android_histogram.c \
				   $(PROJECT_SRC)/android_image_processing.c \
				   $(PROJECT_SRC)/android_image_processing_im2col.c \
				   $(PROJECT_SRC)/android_image_processing_im2row.c \
				   $(PROJECT_SRC)/android_image_processing_gemm.c \
				   $(PROJECT_UTIL_SRC)/android_arm_util.c 

LOCAL_LDLIBS := -lm -llog
LOCAL_LDFLAGS := -nodefaultlibs -lc -lm -ldl

include $(BUILD_EXECUTABLE)
