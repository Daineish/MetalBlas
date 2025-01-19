//
//  metalDot.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

#include <metal_stdlib>
#include "metalReduce.h"
using namespace metal;

template <typename T, int NB>
kernel void metalDotWork(const device int& N [[buffer(0)]],
                       device T* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device T* y[[buffer(3)]],
                       const device int& incy[[buffer(4)]],
                       device T* res[[buffer(5)]],
                       device T* workspace[[buffer(6)]],
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

    threadgroup float s_partial[NB];

    float xvals[32];
    float yvals[32];
    for(int i = 0; i < 32; i++)
    {
        if(gid + i * nt < Nu)
        {
            xvals[i] = x[(gid + i * nt) * incx];
            yvals[i] = y[(gid + i * nt) * incy];
        }
        else
        {
            xvals[i] = yvals[i] = 0;
        }
    }

    for(int i = 0; i < 32; i++)
    {
        s_partial[tid] += (xvals[i] * yvals[i]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint i = NB / 2; i > 0; i /= 2)
    {
        if(tid < i)
            s_partial[tid] += s_partial[tid + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0)
        workspace[tgid] = s_partial[0];
}

template [[host_name("metalSdot")]]
kernel void metalDotWork<float, 1024>(const device int& N [[buffer(0)]],
                       device float* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device float* y[[buffer(3)]],
                       const device int& incy[[buffer(4)]],
                       device float* res[[buffer(5)]],
                       device float* workspace[[buffer(6)]],
                       uint nt [[threads_per_grid]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gsize [[threadgroups_per_grid]],
                       uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalHdot")]]
kernel void metalDotWork<half, 1024>(const device int& N [[buffer(0)]],
                       device half* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device half* y[[buffer(3)]],
                       const device int& incy[[buffer(4)]],
                       device half* res[[buffer(5)]],
                       device half* workspace[[buffer(6)]],
                       uint nt [[threads_per_grid]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gsize [[threadgroups_per_grid]],
                       uint tgsize [[threads_per_threadgroup]]);
