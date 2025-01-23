//
//  metalAmin.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-21.
//

#include <metal_stdlib>
#include "metalReduce.h"
using namespace metal;

template <typename T>
struct PairIdxVal
{
    int idx = 0;
    T val = 0;
};

template <typename T>
constexpr constant T MTL_MAX_VAL = 0;

template <>
constexpr constant float MTL_MAX_VAL<float> = FLT_MAX;

template <>
constexpr constant half MTL_MAX_VAL<half> = HALF_MAX;

template <typename T, int NB>
kernel void metalAminWork(const device int& N [[buffer(0)]],
                       device T* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device int* res[[buffer(3)]],
                       device PairIdxVal<T>* workspace[[buffer(4)]],
                       uint nt [[threads_per_grid]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gsize [[threadgroups_per_grid]],
                       uint tgsize [[threads_per_threadgroup]])
{
    // each threadgroup stores a partial sum at workspace[threadgroupIdx]
    uint Nu = uint(N);
    if(gid >= Nu)
        return;

    if(N <= 0)
    {
        res[0] = -1;
        return;
    }

    threadgroup PairIdxVal<T> s_partial[NB];

    PairIdxVal<T> threadMin;
    threadMin.idx = -1;
    threadMin.val = MTL_MAX_VAL<T>;
    for(int i = 0; i < 32; i++)
    {
        T curMin = fabs(threadMin.val);
        int minIdx = threadMin.idx;
        
        if(gid + i * nt < Nu)
        {
            T curVal = fabs(x[(gid + i * nt) * incx]);
            int curIdx = gid + i * nt;

            if(curVal < curMin || (curVal == curMin && curIdx < minIdx))
            {
                threadMin.idx = curIdx;
                threadMin.val = curVal;
            }
        }
    }

    s_partial[tid] = threadMin;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint i = NB / 2; i > 0; i /= 2)
    {
        if(tid < i && tid + i < Nu)
        {
            T v1 = s_partial[tid].val, v2 = s_partial[tid + i].val;
            int i1 = s_partial[tid].idx, i2 = s_partial[tid + i].idx;
            if(v1 > v2 || (v1 == v2 && i1 > i2))
                s_partial[tid] = s_partial[tid + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0)
        workspace[tgid] = s_partial[0];
}

template [[host_name("metalIsamin")]]
kernel void metalAminWork<float, 1024>(const device int& N [[buffer(0)]],
                       device float* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device int* res[[buffer(3)]],
                       device PairIdxVal<float>* workspace[[buffer(4)]],
                       uint nt [[threads_per_grid]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gsize [[threadgroups_per_grid]],
                       uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalIhamin")]]
kernel void metalAminWork<half, 1024>(const device int& N [[buffer(0)]],
                       device half* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device int* res[[buffer(3)]],
                       device PairIdxVal<half>* workspace[[buffer(4)]],
                       uint nt [[threads_per_grid]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gsize [[threadgroups_per_grid]],
                       uint tgsize [[threads_per_threadgroup]]);
