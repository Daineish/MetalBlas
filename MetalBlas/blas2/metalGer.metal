//
//  metalGer.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalGerWork(const device OrderType& order [[buffer(0)]],
                          const device int& M [[buffer(1)]],
                          const device int& N [[buffer(2)]],
                          const device T& alpha [[buffer(3)]],
                          const device T* x [[buffer(4)]],
                          const device int& incx [[buffer(5)]],
                          const device T* y [[buffer(6)]],
                          const device int& incy [[buffer(7)]],
                          device T* A [[buffer(8)]],
                          const device int& lda [[buffer(9)]],
                          uint gid [[thread_position_in_grid]])
{
    // TODO: prefer short over int where possible
    if(M == 0 || N == 0 || alpha == 0)
        return;

    int rowIdx = gid / N;
    int colIdx = gid % N;

    if(rowIdx >= M || colIdx >= N)
        return;

    const int lda1 = order == ColMajor ? lda : 1;
    const int lda2 = order == ColMajor ? 1 : lda;

    T yVal = alpha * y[colIdx * incy];
    A[colIdx * lda1 + rowIdx * lda2] += yVal * x[rowIdx * incx];
}

template [[host_name("metalHger")]]
kernel void metalGerWork<half>(const device OrderType& order [[buffer(0)]],
                                const device int& M [[buffer(1)]],
                                const device int& N [[buffer(2)]],
                                const device half& alpha [[buffer(3)]],
                                const device half* x [[buffer(4)]],
                                const device int& incx [[buffer(5)]],
                                const device half* y [[buffer(6)]],
                                const device int& incy [[buffer(7)]],
                                device half* A [[buffer(8)]],
                                const device int& lda [[buffer(9)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSger")]]
kernel void metalGerWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device int& M [[buffer(1)]],
                                 const device int& N [[buffer(2)]],
                                 const device float& alpha [[buffer(3)]],
                                 const device float* x [[buffer(4)]],
                                 const device int& incx [[buffer(5)]],
                                 const device float* y [[buffer(6)]],
                                 const device int& incy [[buffer(7)]],
                                 device float* A [[buffer(8)]],
                                 const device int& lda [[buffer(9)]],
                                 uint gid [[thread_position_in_grid]]);
