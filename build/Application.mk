APP_PROJECT_PATH := $(call my-dir)
APP_ABI := arm64-v8a
APP_OPTIM := debug
APP_DEBUG := true

APP_STL := c++_shared # Or system, or none.
APP_CFLAGS := -fsanitize=address -fno-omit-frame-pointer
APP_LDFLAGS := -fsanitize=address

ifeq ($(EN_DEBUG_SYM),true)
APP_STRIP_MODE := none
endif
