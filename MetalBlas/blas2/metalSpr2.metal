//
//  metalSpr2.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-22.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalSpr2Work(const device OrderType& order [[buffer(0)]],
                          const device UploType& uplo [[buffer(1)]],
                          const device int& N [[buffer(2)]],
                          const device T& alpha [[buffer(3)]],
                          const device T* x [[buffer(4)]],
                          const device int& incx [[buffer(5)]],
                          const device T* y [[buffer(6)]],
                          const device int& incy [[buffer(7)]],
                          device T* A [[buffer(8)]],
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
    if(uplo == FillUpper)
    {
        for(int col = gid; col < N; col++)
        {
            int ApIdx = order == ColMajor ? col * (col + 1) / 2 + gid : gid * (N + 1) - (gid * (gid + 1)) / 2 + (col - gid);
            A[ApIdx] += valx * y[col * incy] + valy * x[col * incx];
        }
    }
    else
    {
        for(int col = 0; col <= int(gid); col++)
        {
            int ApIdx = order == ColMajor ? col * (N + 1) - (col * (col + 1)) / 2 + (gid - col) : gid * (gid + 1) / 2 + col;
            A[ApIdx] += valx * y[col * incy] + valy * x[col * incx];
        }
    }
}

template [[host_name("metalHspr2")]]
kernel void metalSpr2Work<half>(const device OrderType& order [[buffer(0)]],
                                const device UploType& uplot [[buffer(1)]],
                                const device int& N [[buffer(2)]],
                                const device half& alpha [[buffer(3)]],
                                const device half* x [[buffer(4)]],
                                const device int& incx [[buffer(5)]],
                                const device half* y [[buffer(6)]],
                                const device int& incy [[buffer(7)]],
                                device half* A [[buffer(8)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSspr2")]]
kernel void metalSpr2Work<float>(const device OrderType& order [[buffer(0)]],
                                 const device UploType& uplo [[buffer(1)]],
                                 const device int& N [[buffer(2)]],
                                 const device float& alpha [[buffer(3)]],
                                 const device float* x [[buffer(4)]],
                                 const device int& incx [[buffer(5)]],
                                 const device float* y [[buffer(6)]],
                                 const device int& incy [[buffer(7)]],
                                 device float* A [[buffer(8)]],
                                 uint gid [[thread_position_in_grid]]);
