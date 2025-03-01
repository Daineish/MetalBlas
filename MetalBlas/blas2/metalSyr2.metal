//
//  metalSyr2.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalSyr2Work(const device OrderType& order [[buffer(0)]],
                          const device UploType& uplo [[buffer(1)]],
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
    if(N == 0 || (alpha == 0))
    {
        return;
    }

    if(gid >= uint(N))
        return;

    T valx = alpha * x[gid * incx];
    T valy = alpha * y[gid * incy];
    const int colMin = uplo == FillUpper ? gid : 0;
    const int colMax = uplo == FillUpper ? N : gid + 1;
    for(int col = colMin; col < colMax; col++)
    {
        int AIdx = order == ColMajor ? col * lda + gid : gid * lda + col;
        A[AIdx] += valx * y[col * incy] + valy * x[col * incx];
    }
}

template [[host_name("metalHsyr2")]]
kernel void metalSyr2Work<half>(const device OrderType& order [[buffer(0)]],
                                const device UploType& uplot [[buffer(1)]],
                                const device int& N [[buffer(2)]],
                                const device half& alpha [[buffer(3)]],
                                const device half* x [[buffer(4)]],
                                const device int& incx [[buffer(5)]],
                                const device half* y [[buffer(6)]],
                                const device int& incy [[buffer(7)]],
                                device half* A [[buffer(8)]],
                                const device int& lda [[buffer(9)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSsyr2")]]
kernel void metalSyr2Work<float>(const device OrderType& order [[buffer(0)]],
                                 const device UploType& uplo [[buffer(1)]],
                                 const device int& N [[buffer(2)]],
                                 const device float& alpha [[buffer(3)]],
                                 const device float* x [[buffer(4)]],
                                 const device int& incx [[buffer(5)]],
                                 const device float* y [[buffer(6)]],
                                 const device int& incy [[buffer(7)]],
                                 device float* A [[buffer(8)]],
                                 const device int& lda [[buffer(9)]],
                                 uint gid [[thread_position_in_grid]]);

