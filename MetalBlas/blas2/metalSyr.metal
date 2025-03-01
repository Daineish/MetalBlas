//
//  metalSyr.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalSyrWork(const device OrderType& order [[buffer(0)]],
                          const device UploType& uplo [[buffer(1)]],
                          const device int& N [[buffer(2)]],
                          const device T& alpha [[buffer(3)]],
                          const device T* x [[buffer(4)]],
                          const device int& incx [[buffer(5)]],
                          device T* A [[buffer(6)]],
                          const device int& lda [[buffer(7)]],
                          uint gid [[thread_position_in_grid]])
{
    if(N == 0 || (alpha == 0))
    {
        return;
    }

    if(gid >= uint(N))
        return;

    T val = alpha * x[gid * incx];

    const int colMin = uplo == FillUpper ? gid : 0;
    const int colMax = uplo == FillUpper ? N : gid + 1;
    for(int col = colMin; col < colMax; col++)
    {
        int AIdx = order == ColMajor ? col * lda + gid : gid * lda + col;
        A[AIdx] += val * x[col * incx];
    }
}

template [[host_name("metalHsyr")]]
kernel void metalSyrWork<half>(const device OrderType& order [[buffer(0)]],
                                const device UploType& uplot [[buffer(1)]],
                                const device int& N [[buffer(2)]],
                                const device half& alpha [[buffer(3)]],
                                const device half* x [[buffer(4)]],
                                const device int& incx [[buffer(5)]],
                                device half* A [[buffer(6)]],
                                const device int& lda [[buffer(7)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSsyr")]]
kernel void metalSyrWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device UploType& uplo [[buffer(1)]],
                                 const device int& N [[buffer(2)]],
                                 const device float& alpha [[buffer(3)]],
                                 const device float* x [[buffer(4)]],
                                 const device int& incx [[buffer(5)]],
                                 device float* A [[buffer(6)]],
                                 const device int& lda [[buffer(7)]],
                                 uint gid [[thread_position_in_grid]]);
