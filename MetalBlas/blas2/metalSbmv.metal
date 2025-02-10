//
//  metalSbmv.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-09.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalSbmvWork(const device OrderType& order [[buffer(0)]],
                          const device UploType& uplo [[buffer(1)]],
                          const device int& N [[buffer(2)]],
                          const device int& K [[buffer(3)]],
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
    if(N == 0 || (alpha == 0 && beta == 1))
        return;

    if(gid > uint(N))
        return;

    T result = 0;
    
    uint minCol = max(0, int(gid) - int(K));
    uint maxCol = min(N, int(gid) + int(K) + 1);
    for(uint curCol = minCol; curCol < maxCol; curCol++)
    {
        uint r = gid;
        uint c = curCol;
        if((uplo == FillUpper && gid > curCol) || (uplo == FillLower && curCol > gid))
        {
            uint tmp = r;
            r = c; c = tmp;
        }

        uint AbIdx = order == RowMajor ? (r * lda + (c - r + (uplo == FillLower ? K : 0))) : (c * lda + (r - c) + (uplo == FillUpper ? K : 0));
        result += A[AbIdx] * x[curCol * incx];
    }

    y[gid * incy] = alpha * result + beta * y[gid * incy];
}

template [[host_name("metalHsbmv")]]
kernel void metalSbmvWork<half>(const device OrderType& order [[buffer(0)]],
                                const device UploType& uplot [[buffer(1)]],
                                const device int& N [[buffer(2)]],
                                const device int& K [[buffer(3)]],
                                const device half& alpha [[buffer(4)]],
                                const device half* A [[buffer(5)]],
                                const device int& lda [[buffer(6)]],
                                const device half* x [[buffer(7)]],
                                const device int& incx [[buffer(8)]],
                                const device half& beta [[buffer(9)]],
                                device half* y [[buffer(10)]],
                                const device int& incy [[buffer(11)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSsbmv")]]
kernel void metalSbmvWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device UploType& uplo [[buffer(1)]],
                                 const device int& N [[buffer(2)]],
                                 const device int& K [[buffer(3)]],
                                 const device float& alpha [[buffer(4)]],
                                 const device float* A [[buffer(5)]],
                                 const device int& lda [[buffer(6)]],
                                 const device float* x [[buffer(7)]],
                                 const device int& incx [[buffer(8)]],
                                 const device float& beta [[buffer(9)]],
                                 device float* y [[buffer(10)]],
                                 const device int& incy [[buffer(11)]],
                                 uint gid [[thread_position_in_grid]]);
