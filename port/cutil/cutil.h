// cutil.h — minimal replacement for the retired NVIDIA GPU Computing SDK
// "cutil" helper library, sufficient to build the ISPASS 2009 benchmark suite
// (ispass2009-benchmarks) with CUDA 12.x on sm_89.
//
// Coverage (inventory taken from LPS / LIB / NQU, Phase 0):
//   Macros      : CUDA_SAFE_CALL, CUDA_SAFE_CALL_NO_SYNC, CUT_SAFE_CALL,
//                 CUT_CHECK_ERROR, CUT_EXIT
//   Cmd line    : cutCheckCmdLineFlag, cutGetCmdLineArgumenti/f/str
//   Timers      : cutCreateTimer/StartTimer/StopTimer/ResetTimer/
//                 GetTimerValue/DeleteTimer/GetTimerCount
//   Removed API : cudaThreadSynchronize -> cudaDeviceSynchronize (CUDA 12)
//
// Semantics note for fault-injection verdicts: on CUDA API error or async
// launch error this shim prints to stderr and exits with EXIT_FAILURE, so a
// run wrapper can classify non-zero exit as CRASH/ERROR instead of
// silently comparing garbage output.

#ifndef CUTIL_SHIM_H
#define CUTIL_SHIM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int CUTBoolean;
#define CUT_TRUE  1
#define CUT_FALSE 0

// ---------------------------------------------------------------------------
// Error handling macros
// ---------------------------------------------------------------------------

static inline void cutilShimFail(const char* what, const char* file, int line)
{
    fprintf(stderr, "[cutil-shim] FAILED %s at %s:%d\n", what, file, line);
    exit(EXIT_FAILURE);
}

static inline void cutSafeCall(int ok, const char* file, int line)
{
    if (!ok) cutilShimFail("CUT_SAFE_CALL", file, line);
}

static inline void cudaSafeCall(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "[cutil-shim] CUDA error at %s:%d: %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static inline void cutCheckError(const char* msg, const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[cutil-shim] %s — CUDA error at %s:%d: %s\n",
                msg ? msg : "", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define CUT_SAFE_CALL(value)        cutSafeCall((int)(value), __FILE__, __LINE__)
#define CUDA_SAFE_CALL(call)        cudaSafeCall((call), __FILE__, __LINE__)
#define CUDA_SAFE_CALL_NO_SYNC(call) cudaSafeCall((call), __FILE__, __LINE__)
#define CUT_CHECK_ERROR(errmsg)     cutCheckError((errmsg), __FILE__, __LINE__)
#define CUT_EXIT(argc, argv)        exit(EXIT_SUCCESS)

// ---------------------------------------------------------------------------
// Command-line parsing
// ---------------------------------------------------------------------------

static inline int cutCheckCmdLineFlag(int argc, const char** argv,
                                      const char* flag)
{
    int i;
    for (i = 1; i < argc; ++i) {
        if (argv[i] && argv[i][0] == '-') {
            const char* a = argv[i] + 1;
            if (a[0] == '-') a++;              /* tolerate "--flag" */
            if (strcmp(a, flag) == 0) return 1;
        }
    }
    return 0;
}

static inline int cutGetCmdLineArgumenti(int argc, const char** argv,
                                         const char* name, int* dest)
{
    int i;
    size_t len = strlen(name);
    for (i = 1; i < argc; ++i) {
        if (argv[i] && argv[i][0] == '-' &&
            strncmp(argv[i] + 1, name, len) == 0 && argv[i][len + 1] == '=') {
            *dest = atoi(argv[i] + len + 2);
            return 1;
        }
    }
    return 0;
}

static inline int cutGetCmdLineArgumentf(int argc, const char** argv,
                                         const char* name, float* dest)
{
    int i;
    size_t len = strlen(name);
    for (i = 1; i < argc; ++i) {
        if (argv[i] && argv[i][0] == '-' &&
            strncmp(argv[i] + 1, name, len) == 0 && argv[i][len + 1] == '=') {
            *dest = (float)atof(argv[i] + len + 2);
            return 1;
        }
    }
    return 0;
}

static inline int cutGetCmdLineArgumentstr(int argc, const char** argv,
                                           const char* name, char** dest)
{
    int i;
    size_t len = strlen(name);
    for (i = 1; i < argc; ++i) {
        if (argv[i] && argv[i][0] == '-' &&
            strncmp(argv[i] + 1, name, len) == 0 && argv[i][len + 1] == '=') {
            *dest = (char*)(argv[i] + len + 2);
            return 1;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Timers (millisecond, monotonic clock)
// ---------------------------------------------------------------------------

#define CUTIL_SHIM_MAX_TIMERS 16

typedef struct {
    struct timespec start;
    float           elapsed_ms;   /* accumulated between Start/Stop */
    int             running;
} cutilShimTimer;

static cutilShimTimer g_cutilTimers[CUTIL_SHIM_MAX_TIMERS];
static int            g_cutilTimerCount = 0;

static inline float cutilShimNowMs(struct timespec t)
{
    return (float)t.tv_sec * 1000.0f + (float)t.tv_nsec / 1.0e6f;
}

static inline CUTBoolean cutCreateTimer(unsigned int* handle)
{
    if (g_cutilTimerCount >= CUTIL_SHIM_MAX_TIMERS) return CUT_FALSE;
    *handle = (unsigned int)g_cutilTimerCount++;
    g_cutilTimers[*handle].running    = 0;
    g_cutilTimers[*handle].elapsed_ms = 0.0f;
    return CUT_TRUE;
}

static inline CUTBoolean cutStartTimer(unsigned int handle)
{
    if (handle >= (unsigned int)g_cutilTimerCount) return CUT_FALSE;
    clock_gettime(CLOCK_MONOTONIC, &g_cutilTimers[handle].start);
    g_cutilTimers[handle].running = 1;
    return CUT_TRUE;
}

static inline CUTBoolean cutStopTimer(unsigned int handle)
{
    struct timespec now;
    if (handle >= (unsigned int)g_cutilTimerCount) return CUT_FALSE;
    if (!g_cutilTimers[handle].running) return CUT_TRUE; // idempotent stop
    clock_gettime(CLOCK_MONOTONIC, &now);
    g_cutilTimers[handle].elapsed_ms +=
        cutilShimNowMs(now) - cutilShimNowMs(g_cutilTimers[handle].start);
    g_cutilTimers[handle].running = 0;
    return CUT_TRUE;
}

static inline CUTBoolean cutResetTimer(unsigned int handle)
{
    if (handle >= (unsigned int)g_cutilTimerCount) return CUT_FALSE;
    g_cutilTimers[handle].elapsed_ms = 0.0f;
    return CUT_TRUE;
}

static inline float cutGetTimerValue(unsigned int handle)
{
    if (handle >= (unsigned int)g_cutilTimerCount) return 0.0f;
    return g_cutilTimers[handle].elapsed_ms;
}

static inline unsigned int cutGetTimerCount(void)
{
    return (unsigned int)g_cutilTimerCount;
}

static inline CUTBoolean cutDeleteTimer(unsigned int handle)
{
    if (handle >= (unsigned int)g_cutilTimerCount) return CUT_FALSE;
    return CUT_TRUE;
}


// Byte-array comparison used by STO (storeGPU) result checking.
inline CUTBoolean cutCompareub(const unsigned char* reference,
                               const unsigned char* data,
                               const unsigned int len) {
    for (unsigned int i = 0; i < len; ++i)
        if (reference[i] != data[i]) return CUT_FALSE;
    return CUT_TRUE;
}

// ---------------------------------------------------------------------------
// CUDA 12 deprecated (removed in 13) API — route to the modern equivalent.
// Macro form avoids conflicting with the still-declared deprecated prototype.
// ---------------------------------------------------------------------------

#define cudaThreadSynchronize() cudaDeviceSynchronize()
// --- shared-memory bank-conflict macro (legacy SDK debug helper) ---
#ifndef CUT_BANK_CHECKER
#define CUT_BANK_CHECKER(symbol, offsets) (symbol)[offsets]
#endif


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUTIL_SHIM_H */
