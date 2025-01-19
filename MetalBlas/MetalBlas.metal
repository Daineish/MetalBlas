//
//  MetalBlas.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-04.
//

#include <metal_stdlib>
using namespace metal;

kernel void metalSgemm(const device int& M [[buffer(0)]],
                          const device int& N [[buffer(1)]],
                          const device int& K [[buffer(2)]],
                          const device float& alpha [[buffer(3)]],
                          const device float* A [[buffer(4)]],
                          const device int& lda [[buffer(5)]],
                          const device float* B [[buffer(6)]],
                          const device int& ldb [[buffer(7)]],
                          const device float& beta [[buffer(8)]],
                          device float* C [[buffer(9)]],
                          const device int& ldc [[buffer(10)]],
                          uint2 gid [[thread_position_in_grid]])
{
    uint globalRow = gid.x;
    uint globalCol = gid.y;
    if(globalRow >= uint(M) || globalCol >= uint(N))
        return;
    float acc = 0.0f;
    for(int i = 0; i < K; i++)
    {
        acc += A[i * lda + globalRow] * B[globalCol * ldb + i];
    }
    C[globalCol * ldc + globalRow] = alpha * acc + beta * C[globalCol * ldc + globalRow];
}

kernel void metalHgemm(const device int& M [[buffer(0)]],
                          const device int& N [[buffer(1)]],
                          const device int& K [[buffer(2)]],
                          const device half& alpha [[buffer(3)]],
                          const device half* A [[buffer(4)]],
                          const device int& lda [[buffer(5)]],
                          const device half* B [[buffer(6)]],
                          const device int& ldb [[buffer(7)]],
                          const device half& beta [[buffer(8)]],
                          device half* C [[buffer(9)]],
                          const device int& ldc [[buffer(10)]],
                          uint2 gid [[thread_position_in_grid]])
{
    uint globalRow = gid.x;
    uint globalCol = gid.y;
    if(globalRow >= uint(M) || globalCol >= uint(N))
        return;
    float acc = 0.0f;
    for(int i = 0; i < K; i++)
    {
        acc += A[i * lda + globalRow] * B[globalCol * ldb + i];
    }
    C[globalCol * ldc + globalRow] = alpha * acc + beta * C[globalCol * ldc + globalRow];
}

kernel void metalSaxpy(const device int& N [[buffer(0)]],
                          const device float& alpha [[buffer(1)]],
                          const device float* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          device float* y [[buffer(4)]],
                          const device int& incy [[buffer(5)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    y[gid * incy] += alpha * x[gid * incx];
}

kernel void metalHaxpy(const device int& N [[buffer(0)]],
                          const device half& alpha [[buffer(1)]],
                          const device half* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          device half* y [[buffer(4)]],
                          const device int& incy [[buffer(5)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    y[gid * incy] += alpha * x[gid * incx];
}

kernel void metalHscal(const device int& N [[buffer(0)]],
                          const device half& alpha [[buffer(1)]],
                          device half* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    x[gid * incx] *= alpha;
}

kernel void metalSscal(const device int& N [[buffer(0)]],
                          const device float& alpha [[buffer(1)]],
                          device float* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    x[gid * incx] *= alpha;
}

template <typename T, int NB>
kernel void metalAsumWork(const device int& N [[buffer(0)]],
                       device T* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device T* res[[buffer(3)]],
                       device T* workspace[[buffer(4)]],
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

    float rvals[32];
    for(int i = 0; i < 32; i++)
    {
        if(gid + i * nt < Nu)
            rvals[i] = abs(x[gid + i * nt]);
        else
            rvals[i] = 0;
    }

    for(int i = 0; i < 32; i++)
    {
        s_partial[tid] += rvals[i];
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

template [[host_name("metalSasum")]]
kernel void metalAsumWork<float, 1024>(const device int& N [[buffer(0)]],
                       device float* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device float* res[[buffer(3)]],
                       device float* workspace[[buffer(4)]],
                       uint nt [[threads_per_grid]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gsize [[threadgroups_per_grid]],
                       uint tgsize [[threads_per_threadgroup]]);

template [[host_name("metalHasum")]]
kernel void metalAsumWork<half, 1024>(const device int& N [[buffer(0)]],
                       device half* x[[buffer(1)]],
                       const device int& incx[[buffer(2)]],
                       device half* res[[buffer(3)]],
                       device half* workspace[[buffer(4)]],
                       uint nt [[threads_per_grid]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint gsize [[threadgroups_per_grid]],
                       uint tgsize [[threads_per_threadgroup]]);

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
