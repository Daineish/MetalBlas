//
//  metalSymv.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-27.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalSymvWork(const device OrderType& order [[buffer(0)]],
                          const device UploType& uplo [[buffer(1)]],
                          const device int& N [[buffer(2)]],
                          const device T& alpha [[buffer(3)]],
                          const device T* A [[buffer(4)]],
                          const device int& lda [[buffer(5)]],
                          const device T* x [[buffer(6)]],
                          const device int& incx [[buffer(7)]],
                          const device T& beta [[buffer(8)]],
                          device T* y [[buffer(9)]],
                          const device int& incy [[buffer(10)]],
                          uint gid [[thread_position_in_grid]])
{
    if(N == 0 || (alpha == 0 && beta == 1))
        return;

    if(gid >= uint(N))
        return;

    T sum = 0;
    for(int col = 0; col < N; col++)
    {
        int r = gid;
        int c = col;

        if((uplo == FillUpper && gid > uint(col)) || (uplo == FillLower && gid < uint(col)))
        {
            int tmp = r;
            r = c; c = tmp;
        }

        int Aidx = order == ColMajor ? c * lda + r : r * lda + c;
        sum += A[Aidx] * x[col * incx];
    }

    y[gid * incy] = alpha * sum + beta * y[gid * incy];
}

template [[host_name("metalHsymv")]]
kernel void metalSymvWork<half>(const device OrderType& order [[buffer(0)]],
                                const device UploType& uplot [[buffer(1)]],
                                const device int& N [[buffer(2)]],
                                const device half& alpha [[buffer(3)]],
                                const device half* A [[buffer(4)]],
                                const device int& lda [[buffer(5)]],
                                const device half* x [[buffer(6)]],
                                const device int& incx [[buffer(7)]],
                                const device half& beta [[buffer(8)]],
                                device half* y [[buffer(9)]],
                                const device int& incy [[buffer(10)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSsymv")]]
kernel void metalSymvWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device UploType& uplo [[buffer(1)]],
                                 const device int& N [[buffer(2)]],
                                 const device float& alpha [[buffer(3)]],
                                 const device float* A [[buffer(4)]],
                                 const device int& lda [[buffer(5)]],
                                 const device float* x [[buffer(6)]],
                                 const device int& incx [[buffer(7)]],
                                 const device float& beta [[buffer(8)]],
                                 device float* y [[buffer(9)]],
                                 const device int& incy [[buffer(10)]],
                                 uint gid [[thread_position_in_grid]]);
