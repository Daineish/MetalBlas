//
//  metalReduce.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

#include <metal_stdlib>
using namespace metal;


template <typename T>
kernel void metalsthreadgroupReduce(const device int& N [[buffer(0)]],
                                    device T* w[[buffer(1)]],
                              device T* res[[buffer(2)]],
                              uint gid [[thread_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                              uint tgsize [[threads_per_threadgroup]])
{
    uint Nu = N;
    if(tid >= Nu)
        return;

    threadgroup T s_partial[1024];
    if(tid < Nu)
        s_partial[tid] = abs(w[tid]);
    else
        s_partial[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // TODO: 1024 to next_po2(N) ?
    for(uint i = 1024 / 2; i > 0; i /= 2)
    {
        if(tid < i)
            s_partial[tid] += s_partial[tid + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0)
        res[0] = s_partial[0];
}

template [[host_name("metalsReduce")]]
kernel void metalsthreadgroupReduce<float>(const device int& N [[buffer(0)]],
                                           device float* w[[buffer(1)]],
                                           device float* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalhReduce")]]
kernel void metalsthreadgroupReduce<half>(const device int& N [[buffer(0)]],
                                           device half* w[[buffer(1)]],
                                           device half* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

