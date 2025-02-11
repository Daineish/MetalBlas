//
//  metalSpmv.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-09.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalSpmvWork(const device OrderType& order [[buffer(0)]],
                          const device UploType& uplo [[buffer(1)]],
                          const device int& N [[buffer(2)]],
                          const device T& alpha [[buffer(3)]],
                          const device T* A [[buffer(4)]],
                          const device T* x [[buffer(5)]],
                          const device int& incx [[buffer(6)]],
                          const device T& beta [[buffer(7)]],
                          device T* y [[buffer(8)]],
                          const device int& incy [[buffer(9)]],
                          uint gid [[thread_position_in_grid]])
{
    if(N == 0 || (alpha == 0 && beta == 1))
        return;

    if(gid > uint(N))
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

        if(uplo == FillUpper)
        {
            int ApIdx = order == ColMajor ? c * (c + 1) / 2 + r : r * (N + 1) - (r * (r + 1)) / 2 + (c - r);
            sum += A[ApIdx] * x[col * incx];
        }
        else
        {
            int ApIdx = order == ColMajor ? c * (N + 1) - (c * (c + 1)) / 2 + (r - c) : r * (r + 1) / 2 + c;
            sum += A[ApIdx] * x[col * incx];
        }
    }

    y[gid * incy] = alpha * sum + beta * y[gid * incy];
}

template [[host_name("metalHspmv")]]
kernel void metalSpmvWork<half>(const device OrderType& order [[buffer(0)]],
                                const device UploType& uplot [[buffer(1)]],
                                const device int& N [[buffer(2)]],
                                const device half& alpha [[buffer(3)]],
                                const device half* A [[buffer(4)]],
                                const device half* x [[buffer(5)]],
                                const device int& incx [[buffer(6)]],
                                const device half& beta [[buffer(7)]],
                                device half* y [[buffer(8)]],
                                const device int& incy [[buffer(9)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSspmv")]]
kernel void metalSpmvWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device UploType& uplo [[buffer(1)]],
                                 const device int& N [[buffer(2)]],
                                 const device float& alpha [[buffer(3)]],
                                 const device float* A [[buffer(4)]],
                                 const device float* x [[buffer(5)]],
                                 const device int& incx [[buffer(6)]],
                                 const device float& beta [[buffer(7)]],
                                 device float* y [[buffer(8)]],
                                 const device int& incy [[buffer(9)]],
                                 uint gid [[thread_position_in_grid]]);
