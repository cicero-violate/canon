// Compatibility shim: CUDA 11.8 + glibc 2.41
// glibc 2.41 added noexcept(true) to cospi/sinpi/rsqrt family,
// but CUDA 11.8 math_functions.h declares them without it.
// Pre-declaring with noexcept(true) here (before glibc headers fire)
// satisfies the C++ requirement that all declarations match.
#pragma once
#ifndef __CUDA_MATH_COMPAT_H__
#define __CUDA_MATH_COMPAT_H__

// Only apply when glibc >= 2.41 introduces these noexcept specs
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 41))

// Forward-declare with noexcept(true) before CUDA or glibc headers see them.
// These match the glibc 2.41 declarations so CUDA's later re-declaration is consistent.
extern double cospi(double x) noexcept(true);
extern double sinpi(double x) noexcept(true);
extern float  cospif(float x) noexcept(true);
extern float  sinpif(float x) noexcept(true);
extern double rsqrt(double x) noexcept(true);
extern float  rsqrtf(float x) noexcept(true);

#endif // glibc >= 2.41
#endif // __CUDA_MATH_COMPAT_H__
