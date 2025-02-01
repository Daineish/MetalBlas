//
//  metalGemv.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalGemvWork(const device OrderType& order [[buffer(0)]],
                          const device TransposeType& trans [[buffer(1)]],
                          const device int& M [[buffer(2)]],
                          const device int& N [[buffer(3)]],
                          const device T& alpha [[buffer(4)]],
                          const device T* A [[buffer(5)]],
                          const device int& lda [[buffer(6)]],
                          const device T* x [[buffer(7)]],
                          const device int& incx [[buffer(8)]],
                          const device T& beta [[buffer(9)]],
                          device T* y [[buffer(10)]],
                          const device int& incy [[buffer(11)]],
                          uint gid [[thread_position_in_grid]])
{
    if(M == 0 || N == 0 || (alpha == 0 && beta == 1))
        return;

    uint globalRow = gid;

    int lda1 = order == ColMajor ? lda : 1;
    int lda2 = order == ColMajor ? 1 : lda;
    int K = trans == NoTranspose ? N : M;
    int K1 = trans == NoTranspose ? M : N;

    if(globalRow >= uint(K1))
        return;

    float acc = 0.0;
    for(int i = 0; i < K; i++)
    {
        int Aidx = trans == NoTranspose ? i * lda1 + globalRow * lda2 : globalRow * lda1 + i * lda2;
        acc += A[Aidx] * x[i * incx];
    }

    y[globalRow * incy] = alpha * acc + beta * y[globalRow * incy];
}

template [[host_name("metalHgemv")]]
kernel void metalGemvWork<half>(const device OrderType& order [[buffer(0)]],
                                const device TransposeType& trans [[buffer(1)]],
                                const device int& M [[buffer(2)]],
                                const device int& N [[buffer(3)]],
                                const device half& alpha [[buffer(4)]],
                                const device half* A [[buffer(5)]],
                                const device int& lda [[buffer(6)]],
                                const device half* x [[buffer(7)]],
                                const device int& incx [[buffer(8)]],
                                const device half& beta [[buffer(9)]],
                                device half* y [[buffer(10)]],
                                const device int& incy [[buffer(11)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSgemv")]]
kernel void metalGemvWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device TransposeType& trans [[buffer(1)]],
                                 const device int& M [[buffer(2)]],
                                 const device int& N [[buffer(3)]],
                                 const device float& alpha [[buffer(4)]],
                                 const device float* A [[buffer(5)]],
                                 const device int& lda [[buffer(6)]],
                                 const device float* x [[buffer(7)]],
                                 const device int& incx [[buffer(8)]],
                                 const device float& beta [[buffer(9)]],
                                 device float* y [[buffer(10)]],
                                 const device int& incy [[buffer(11)]],
                                 uint gid [[thread_position_in_grid]]);
