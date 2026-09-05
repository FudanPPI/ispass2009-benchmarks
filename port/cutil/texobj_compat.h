// texobj_compat.h — replacement for the legacy texture-reference API
// (texture<...> template class + cudaBindTexture*), which CUDA 12.9 rejects.
// Device-side texture references become `__device__ cudaTextureObject_t`;
// host-side binding creates a texture object and copies the handle into the
// symbol. Fetch sites keep their syntax but need an explicit type argument:
//   tex1Dfetch<float>(tex, i) / tex2D<ulong4>(tex, x, y)
#ifndef TEXOBJ_COMPAT_H
#define TEXOBJ_COMPAT_H

#include <cuda_runtime.h>
#include <string.h>

// NOTE: the pointer parameter must not be named `devPtr` — that name also
// appears as the struct member `_rd.res.linear.devPtr` and would be
// textually replaced by the argument.
#define TEXOBJ_CREATE_1D(obj, texPtr, szBytes, channeldesc)                    \
    do {                                                                       \
        cudaResourceDesc _rd;                                                  \
        memset(&_rd, 0, sizeof(_rd));                                          \
        _rd.resType = cudaResourceTypeLinear;                                  \
        _rd.res.linear.devPtr = (void*)(texPtr);                               \
        _rd.res.linear.desc = (channeldesc);                                   \
        _rd.res.linear.sizeInBytes = (size_t)(szBytes);                        \
        cudaTextureDesc _td;                                                   \
        memset(&_td, 0, sizeof(_td));                                          \
        _td.addressMode[0] = cudaAddressModeClamp;                             \
        _td.addressMode[1] = cudaAddressModeClamp;                             \
        _td.filterMode = cudaFilterModePoint;                                  \
        _td.readMode = cudaReadModeElementType;                                \
        _td.normalizedCoords = 0;                                              \
        (obj) = 0;                                                             \
        cudaCreateTextureObject(&(obj), &_rd, &_td, NULL);                     \
    } while (0)

#define TEXOBJ_CREATE_ARRAY(obj, arr, channeldesc)                             \
    do {                                                                       \
        cudaResourceDesc _rd;                                                  \
        memset(&_rd, 0, sizeof(_rd));                                          \
        _rd.resType = cudaResourceTypeArray;                                   \
        _rd.res.array.array = (cudaArray_t)(arr);                              \
        cudaTextureDesc _td;                                                   \
        memset(&_td, 0, sizeof(_td));                                          \
        _td.addressMode[0] = cudaAddressModeClamp;                             \
        _td.addressMode[1] = cudaAddressModeClamp;                             \
        _td.filterMode = cudaFilterModePoint;                                  \
        _td.readMode = cudaReadModeElementType;                                \
        _td.normalizedCoords = 0;                                              \
        (obj) = 0;                                                             \
        cudaCreateTextureObject(&(obj), &_rd, &_td, NULL);                     \
    } while (0)

// Bind a 1D linear-memory texture (replaces cudaBindTexture(0, sym, ptr, sz))
#define BIND_TEX1D(sym, devPtr, szBytes, channeldesc)                          \
    do {                                                                       \
        cudaTextureObject_t _obj;                                              \
        TEXOBJ_CREATE_1D(_obj, (devPtr), (szBytes), (channeldesc));            \
        cudaMemcpyToSymbol(sym, &_obj, sizeof(_obj));                          \
    } while (0)

// Bind a 2D cudaArray-backed texture (replaces cudaBindTextureToArray)
#define BIND_TEX_ARRAY(sym, arr, channeldesc)                                  \
    do {                                                                       \
        cudaTextureObject_t _obj;                                              \
        TEXOBJ_CREATE_ARRAY(_obj, (arr), (channeldesc));                       \
        cudaMemcpyToSymbol(sym, &_obj, sizeof(_obj));                          \
    } while (0)

// Replace cudaUnbindTexture: destroy the current object handle so repeated
// bind cycles don't leak objects.
#define UNBIND_TEX(sym)                                                        \
    do {                                                                       \
        void* _sa = NULL;                                                      \
        if (cudaGetSymbolAddress(&_sa, sym) == cudaSuccess && _sa) {           \
            cudaTextureObject_t _old = 0;                                      \
            cudaMemcpy(&_old, _sa, sizeof(_old), cudaMemcpyDeviceToHost);      \
            if (_old) cudaDestroyTextureObject(_old);                          \
        }                                                                      \
    } while (0)

#endif /* TEXOBJ_COMPAT_H */
