//
//  metalReduce.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

#include <metal_stdlib>
using namespace metal;


template <typename T, template <typename> class FINALIZE>
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
        s_partial[tid] = w[tid];
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
        res[0] = FINALIZE<T>::call(s_partial[0]);
}

template <typename T>
kernel void mtlSqrt(device T* res [[buffer(0)]])
{
    return sqrt(res);
}

kernel void myHelper()
{
    return;
}

template <typename T>
struct FinalizeSqrt
{
    static T call(T a)
    {
        return sqrt(a);
    }
};

template <typename T>
struct FinalizeNop
{
    static T call(T a)
    {
        return a;
    }
};

template [[host_name("metalsReduceSq")]]
kernel void metalsthreadgroupReduce<float, FinalizeSqrt>(const device int& N [[buffer(0)]],
                                           device float* w[[buffer(1)]],
                                           device float* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalhReduceSq")]]
kernel void metalsthreadgroupReduce<half, FinalizeSqrt>(const device int& N [[buffer(0)]],
                                           device half* w[[buffer(1)]],
                                           device half* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalsReduce")]]
kernel void metalsthreadgroupReduce<float, FinalizeNop>(const device int& N [[buffer(0)]],
                                           device float* w[[buffer(1)]],
                                           device float* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalhReduce")]]
kernel void metalsthreadgroupReduce<half, FinalizeNop>(const device int& N [[buffer(0)]],
                                           device half* w[[buffer(1)]],
                                           device half* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

