//
//  metalReduce.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

#include <metal_stdlib>
using namespace metal;

template <typename T>
struct PairIdxVal
{
    int idx = 0;
    T val = 0;
};

template <typename T, typename W, template <typename> class OPER, template <typename> class FINALIZE>
kernel void metalsthreadgroupReduce(const device int& N [[buffer(0)]],
                                    device W* w[[buffer(1)]],
                              device T* res[[buffer(2)]],
                              uint gid [[thread_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                              uint tgsize [[threads_per_threadgroup]])
{
    uint Nu = N;
    if(tid >= Nu)
        return;

    threadgroup W s_partial[1024];
    if(tid < Nu)
        s_partial[tid] = w[tid];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // TODO: 1024 to next_po2(N) ?
    for(uint i = 1024 / 2; i > 0; i /= 2)
    {
        if(tid < i && tid + i < Nu)
            s_partial[tid] = OPER<W>::call(s_partial[tid], s_partial[tid + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid == 0)
        res[0] = FINALIZE<W>::call(s_partial[0]);
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

template <typename T>
struct FinalizeIdx
{
    static int call(T a)
    {
        return a.idx;
    }
};

template <typename T>
struct OperationAdd
{
    static T call(T a, T b)
    {
        return a + b;
    }
};

template <typename T>
struct OperationMin
{
    static T call(T a, T b)
    {
        if(a.val < b.val || (a.val == b.val && a.idx < b.idx))
            return a;
        else
            return b;
    }
};

template <typename T>
struct OperationMax
{
    static T call(T a, T b)
    {
        if(a.val > b.val || (a.val == b.val && a.idx < b.idx))
            return a;
        else
            return b;
    }
};

template [[host_name("metalsReduceSq")]]
kernel void metalsthreadgroupReduce<float, float, OperationAdd, FinalizeSqrt>(const device int& N [[buffer(0)]],
                                           device float* w[[buffer(1)]],
                                           device float* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalhReduceSq")]]
kernel void metalsthreadgroupReduce<half, half, OperationAdd, FinalizeSqrt>(const device int& N [[buffer(0)]],
                                           device half* w[[buffer(1)]],
                                           device half* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalsReduceMin")]]
kernel void metalsthreadgroupReduce<int, PairIdxVal<float>, OperationMin, FinalizeIdx>(const device int& N [[buffer(0)]],
                                           device PairIdxVal<float>* w[[buffer(1)]],
                                           device int* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalhReduceMin")]]
kernel void metalsthreadgroupReduce<int, PairIdxVal<half>, OperationMin, FinalizeIdx>(const device int& N [[buffer(0)]],
                                           device PairIdxVal<half>* w[[buffer(1)]],
                                           device int* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalsReduceMax")]]
kernel void metalsthreadgroupReduce<int, PairIdxVal<float>, OperationMax, FinalizeIdx>(const device int& N [[buffer(0)]],
                                           device PairIdxVal<float>* w[[buffer(1)]],
                                           device int* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalhReduceMax")]]
kernel void metalsthreadgroupReduce<int, PairIdxVal<half>, OperationMax, FinalizeIdx>(const device int& N [[buffer(0)]],
                                           device PairIdxVal<half>* w[[buffer(1)]],
                                           device int* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalsReduce")]]
kernel void metalsthreadgroupReduce<float, float, OperationAdd, FinalizeNop>(const device int& N [[buffer(0)]],
                                           device float* w[[buffer(1)]],
                                           device float* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalhReduce")]]
kernel void metalsthreadgroupReduce<half, half, OperationAdd, FinalizeNop>(const device int& N [[buffer(0)]],
                                           device half* w[[buffer(1)]],
                                           device half* res[[buffer(2)]],
                                           uint gid [[thread_position_in_grid]],
                                           uint tid [[thread_position_in_threadgroup]],
                                           uint tgsize [[threads_per_threadgroup]]);

