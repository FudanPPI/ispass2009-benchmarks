// cutil_math.h — minimal CUDA SDK cutil_math shim for ISPASS-2009 ports.
// Covers only what the ported apps actually use (RAY: float3/float4 arithmetic,
// cross/dot/normalize; vector types come from CUDA's vector_types.h).
#ifndef CUTIL_MATH_SHIM_H
#define CUTIL_MATH_SHIM_H

#include <math.h>
#include <vector_types.h>
#include <vector_functions.h>

static __host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

static __host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __host__ __device__ inline float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static __host__ __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

static __host__ __device__ inline float3 normalize(const float3& v) {
    float inv = 1.0f / sqrtf(dot(v, v));
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

// --- cross-size constructors (SDK cutil_math provides these) ---
static __host__ __device__ inline float3 make_float3(const float4& v) { return make_float3(v.x, v.y, v.z); }
static __host__ __device__ inline float4 make_float4(const float3& v, float w = 0.0f) { return make_float4(v.x, v.y, v.z, w); }
static __host__ __device__ inline float2 make_float2(const float3& v) { return make_float2(v.x, v.y); }
static __host__ __device__ inline float3 make_float3(const float2& v, float z = 0.0f) { return make_float3(v.x, v.y, z); }

// --- binary operators: vector OP vector ---
static __host__ __device__ inline float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __host__ __device__ inline float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __host__ __device__ inline float3 operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __host__ __device__ inline float3 operator/(const float3& a, const float3& b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
static __host__ __device__ inline float4 operator+(const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __host__ __device__ inline float4 operator-(const float4& a, const float4& b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __host__ __device__ inline float4 operator*(const float4& a, const float4& b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }

// --- binary operators: vector OP scalar (and scalar OP vector) ---
static __host__ __device__ inline float3 operator*(const float3& a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
static __host__ __device__ inline float3 operator*(float s, const float3& a) { return make_float3(a.x * s, a.y * s, a.z * s); }
static __host__ __device__ inline float3 operator/(const float3& a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
static __host__ __device__ inline float3 operator+(const float3& a, float s) { return make_float3(a.x + s, a.y + s, a.z + s); }
static __host__ __device__ inline float3 operator-(const float3& a, float s) { return make_float3(a.x - s, a.y - s, a.z - s); }
static __host__ __device__ inline float4 operator*(const float4& a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
static __host__ __device__ inline float4 operator*(float s, const float4& a) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
static __host__ __device__ inline float4 operator/(const float4& a, float s) { return make_float4(a.x / s, a.y / s, a.z / s, a.w / s); }

// --- unary minus / compound assignment ---
static __host__ __device__ inline float3 operator-(const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
static __host__ __device__ inline float4 operator-(const float4& a) { return make_float4(-a.x, -a.y, -a.z, -a.w); }
static __host__ __device__ inline float3& operator+=(float3& a, const float3& b) { a = a + b; return a; }
static __host__ __device__ inline float3& operator-=(float3& a, const float3& b) { a = a - b; return a; }
static __host__ __device__ inline float3& operator*=(float3& a, float s) { a = a * s; return a; }
static __host__ __device__ inline float4& operator+=(float4& a, const float4& b) { a = a + b; return a; }
static __host__ __device__ inline float4& operator*=(float4& a, float s) { a = a * s; return a; }

#endif /* CUTIL_MATH_SHIM_H */
