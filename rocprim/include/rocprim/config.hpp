// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_CONFIG_HPP_
#define ROCPRIM_CONFIG_HPP_

#include <limits>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "rocprim_version.hpp"

// Inline namespace (e.g. ROCPRIM_300400_NS where 300400 is the rocPRIM version) is used to
// eliminate issues when shared libraries are built with different versions of rocPRIM so they may
// have symbols with the same name but different content.
// ROCPRIM_DISABLE_INLINE_NAMESPACE can be defined to disable inline namespaces (the old behavior).
// ROCPRIM_INLINE_NAMESPACE can be defined to override the standard inline namespace name.
// Additionally, all kernels have hidden visibility (see ROCPRIM_KERNEL).
#if defined(DOXYGEN_DOCUMENTATION_BUILD) || defined(ROCPRIM_DISABLE_INLINE_NAMESPACE)
    #define ROCPRIM_INLINE_NAMESPACE
    #define BEGIN_ROCPRIM_INLINE_NAMESPACE
    #define END_ROCPRIM_INLINE_NAMESPACE
#else
    #define ROCPRIM_CONCAT_(SEP, A, B) A##SEP##B
    #define ROCPRIM_CONCAT(SEP, A, B) ROCPRIM_CONCAT_(SEP, A, B)

    #ifndef ROCPRIM_INLINE_NAMESPACE
        #define ROCPRIM_INLINE_NAMESPACE \
            ROCPRIM_CONCAT(_, ROCPRIM, ROCPRIM_CONCAT(_, ROCPRIM_VERSION, NS))
    #endif
    #define BEGIN_ROCPRIM_INLINE_NAMESPACE        \
        inline namespace ROCPRIM_INLINE_NAMESPACE \
        {
    #define END_ROCPRIM_INLINE_NAMESPACE } /* inline namespace */
#endif

#define BEGIN_ROCPRIM_NAMESPACE \
    namespace rocprim           \
    {                           \
    BEGIN_ROCPRIM_INLINE_NAMESPACE

#define END_ROCPRIM_NAMESPACE    \
    END_ROCPRIM_INLINE_NAMESPACE \
    } /* namespace rocprim */

#if __cplusplus == 201402L
    #warning "rocPRIM C++14 will be deprecated in the next major release"
#endif

#if __cplusplus < 201402L
    #error "rocPRIM requires at least C++14"
#endif

#if !defined(ROCPRIM_DEVICE) || defined(DOXYGEN_DOCUMENTATION_BUILD)
    #define ROCPRIM_DEVICE __device__
    #define ROCPRIM_HOST __host__
    #define ROCPRIM_HOST_DEVICE __host__ __device__
    #define ROCPRIM_SHARED_MEMORY __shared__
    // TODO: These parameters should be tuned for NAVI in the close future.
    #ifndef ROCPRIM_DEFAULT_MAX_BLOCK_SIZE
        #define ROCPRIM_DEFAULT_MAX_BLOCK_SIZE 256
    #endif
    #ifndef ROCPRIM_DEFAULT_MIN_WARPS_PER_EU
        #define ROCPRIM_DEFAULT_MIN_WARPS_PER_EU 1
    #endif

    #ifndef DOXYGEN_DOCUMENTATION_BUILD
        #define ROCPRIM_FORCE_INLINE __attribute__((always_inline))
        #define ROCPRIM_INLINE inline
        #define ROCPRIM_KERNEL __global__ __attribute__((__visibility__("hidden")))
        #define ROCPRIM_LAUNCH_BOUNDS(...) __launch_bounds__(__VA_ARGS__)
    #else
        // Prefer simpler signatures to let Sphinx/Breathe parse them
        #define ROCPRIM_FORCE_INLINE inline
        #define ROCPRIM_INLINE inline
        #define ROCPRIM_KERNEL __global__
        // Ignore __launch_bounds__ for doxygen builds
        #define ROCPRIM_LAUNCH_BOUNDS(...)
    #endif
#endif

#undef ROCPRIM_TARGET_GCN3
#undef ROCPRIM_TARGET_GCN5
#undef ROCPRIM_TARGET_RDNA1
#undef ROCPRIM_TARGET_RDNA2
#undef ROCPRIM_TARGET_RDNA3
#undef ROCPRIM_TARGET_RDNA4
#undef ROCPRIM_TARGET_CDNA1
#undef ROCPRIM_TARGET_CDNA2
#undef ROCPRIM_TARGET_CDNA3
#undef ROCPRIM_TARGET_CDNA4

// See https://llvm.org/docs/AMDGPUUsage.html#instructions
#if defined(__gfx950__)
    #define ROCPRIM_TARGET_CDNA4 1
#elif defined(__gfx942__)
    #define ROCPRIM_TARGET_CDNA3 1
#elif defined(__gfx90a__)
    #define ROCPRIM_TARGET_CDNA2 1
#elif defined(__gfx908__)
    #define ROCPRIM_TARGET_CDNA1 1
#elif defined(__gfx900__) || defined(__gfx902__) || defined(__gfx904__) || defined(__gfx906__) \
    || defined(__gfx90c__)
    #define ROCPRIM_TARGET_GCN5 1
#elif defined(__GFX12__)
    #define ROCPRIM_TARGET_RDNA4 1
#elif defined(__GFX11__)
    #define ROCPRIM_TARGET_RDNA3 1
#elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) \
    || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
    #define ROCPRIM_TARGET_RDNA2 1
#elif defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__)
    #define ROCPRIM_TARGET_RDNA1 1
#elif defined(__GFX8__)
    #define ROCPRIM_TARGET_GCN3 1
#endif

// DPP is supported only after Volcanic Islands (GFX8+)
// Only defined when support is present, in contrast to ROCPRIM_DETAIL_USE_DPP, which should be
// always defined
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) \
    && (!defined(__GFX6__) && !defined(__GFX7__))
    #define ROCPRIM_DETAIL_HAS_DPP 1
#endif

#if !defined(ROCPRIM_DISABLE_DPP) && defined(ROCPRIM_DETAIL_HAS_DPP)
    #define ROCPRIM_DETAIL_USE_DPP 1
#else
    #define ROCPRIM_DETAIL_USE_DPP 0
#endif

#if defined(ROCPRIM_DETAIL_HAS_DPP) && (defined(__GFX8__) || defined(__GFX9__))
    #define ROCPRIM_DETAIL_HAS_DPP_BROADCAST 1
#endif

#if defined(ROCPRIM_DETAIL_HAS_DPP) && (defined(__GFX8__) || defined(__GFX9__))
    #define ROCPRIM_DETAIL_HAS_DPP_WF 1
#endif

#ifndef ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS
    #define ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS 1
#endif

#ifndef ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS
    #define ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS 1
#endif

#ifndef ROCPRIM_NAVI
    #if defined(__HIP_DEVICE_COMPILE__) \
        && (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
        #define ROCPRIM_NAVI 1
    #else
        #define ROCPRIM_NAVI 0
    #endif
#endif
#define ROCPRIM_ARCH_90a 910

/// Supported warp sizes
#define ROCPRIM_WARP_SIZE_32 32u
#define ROCPRIM_WARP_SIZE_64 64u
#define ROCPRIM_MAX_WARP_SIZE ROCPRIM_WARP_SIZE_64

/// Quad size (group of 4 threads)
#define ROCPRIM_QUAD_SIZE 4u

#if(defined(_MSC_VER) && !defined(__clang__)) || (defined(__GNUC__) && !defined(__clang__))
    #define ROCPRIM_UNROLL
    #define ROCPRIM_NO_UNROLL
#else
    #define ROCPRIM_UNROLL _Pragma("unroll")
    #define ROCPRIM_NO_UNROLL _Pragma("nounroll")
#endif

#ifndef ROCPRIM_GRID_SIZE_LIMIT
    #define ROCPRIM_GRID_SIZE_LIMIT std::numeric_limits<unsigned int>::max()
#endif

#if __cpp_if_constexpr >= 201606
    #define ROCPRIM_IF_CONSTEXPR constexpr
#else
    #define ROCPRIM_IF_CONSTEXPR
#endif

//  Copyright 2001 John Maddock.
//  Copyright 2017 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See http://www.boost.org/LICENSE_1_0.txt
//
//  BOOST_STRINGIZE(X)
#define ROCPRIM_STRINGIZE(X) ROCPRIM_DO_STRINGIZE(X)
#define ROCPRIM_DO_STRINGIZE(X) #X

//  Copyright 2017 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See http://www.boost.org/LICENSE_1_0.txt
//
//  BOOST_PRAGMA_MESSAGE("message")
//
//  Expands to the equivalent of #pragma message("message")
#if defined(__INTEL_COMPILER)
    #define ROCPRIM_PRAGMA_MESSAGE(x) \
        __pragma(message(__FILE__ "(" ROCPRIM_STRINGIZE(__LINE__) "): note: " x))
#elif defined(__GNUC__)
    #define ROCPRIM_PRAGMA_MESSAGE(x) _Pragma(ROCPRIM_STRINGIZE(message(x)))
#elif defined(_MSC_VER)
    #define ROCPRIM_PRAGMA_MESSAGE(x) \
        __pragma(message(__FILE__ "(" ROCPRIM_STRINGIZE(__LINE__) "): note: " x))
#else
    #define ROCPRIM_PRAGMA_MESSAGE(x)
#endif

/// Static asserts that only gets evaluated during device compile.
#if __HIP_DEVICE_COMPILE__
    #define ROCPRIM_DETAIL_DEVICE_STATIC_ASSERT(expression, message) \
        static_assert(expression, message)
#else
    #define ROCPRIM_DETAIL_DEVICE_STATIC_ASSERT(expression, message)
#endif

/// \brief Clang predefined macro for device code on AMD GPU targets, either 32 or 64.
///   It is undefined behavior to use this macro in host code when compiling with Clang.
#ifndef __AMDGCN_WAVEFRONT_SIZE
    #define __AMDGCN_WAVEFRONT_SIZE 64
#endif

/// \brief Wavefront size, either 32 or 64. May be defined by compiler flags when compiling
///   with Clang if the value is equal to the wavefront size of all AMD GPU architectures
///   currently being compiled for.
///
///   Only defined in device code unless defined by compiler flags as described above.
#ifndef ROCPRIM_WAVEFRONT_SIZE
    #define ROCPRIM_WAVEFRONT_SIZE __AMDGCN_WAVEFRONT_SIZE
#endif

// Helper macros to disable warnings in clang
#ifdef __clang__
    #define ROCPRIM_PRAGMA_TO_STR(x) _Pragma(#x)
    #define ROCPRIM_CLANG_SUPPRESS_WARNING_PUSH _Pragma("clang diagnostic push")
    #define ROCPRIM_CLANG_SUPPRESS_WARNING(w) ROCPRIM_PRAGMA_TO_STR(clang diagnostic ignored w)
    #define ROCPRIM_CLANG_SUPPRESS_WARNING_POP _Pragma("clang diagnostic pop")
    #define ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH(w) \
        ROCPRIM_CLANG_SUPPRESS_WARNING_PUSH ROCPRIM_CLANG_SUPPRESS_WARNING(w)
#else // __clang__
    #define ROCPRIM_CLANG_SUPPRESS_WARNING_PUSH
    #define ROCPRIM_CLANG_SUPPRESS_WARNING(w)
    #define ROCPRIM_CLANG_SUPPRESS_WARNING_POP
    #define ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH(w)
#endif // __clang__

#ifdef ROCPRIM_DONT_SUPPRESS_DEPRECATIONS
    #define ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH
    #define ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP
#else
    /// \brief Disables warnings from using deprecated symbols until the next
    /// `ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP`.
    /// \note Track the usage of deprecated symbols in the project's issue tracker!
    #define ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_WITH_PUSH \
        ROCPRIM_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated-declarations")
    #define ROCPRIM_DETAIL_SUPPRESS_DEPRECATION_POP ROCPRIM_CLANG_SUPPRESS_WARNING_POP
#endif

#endif // ROCPRIM_CONFIG_HPP_
