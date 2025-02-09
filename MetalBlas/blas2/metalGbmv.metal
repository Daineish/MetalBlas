//
//  metalGbmv.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-03.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalGbmvWork(const device OrderType& order [[buffer(0)]],
                          const device TransposeType& trans [[buffer(1)]],
                          const device int& M [[buffer(2)]],
                          const device int& N [[buffer(3)]],
                          const device int& KL [[buffer(4)]],
                          const device int& KU [[buffer(5)]],
                          const device T& alpha [[buffer(6)]],
                          const device T* A [[buffer(7)]],
                          const device int& lda [[buffer(8)]],
                          const device T* x [[buffer(9)]],
                          const device int& incx [[buffer(10)]],
                          const device T& beta [[buffer(11)]],
                          device T* y [[buffer(12)]],
                          const device int& incy [[buffer(13)]],
                          uint gid [[thread_position_in_grid]])
{
    if(M == 0 || N == 0 || (alpha == 0 && beta == 1))
        return;

    if(gid > uint(trans == NoTranspose ? M : N))
        return;

    T result = 0;
    
    if(trans == NoTranspose)
    {
        uint minCol = max(0, int(gid) - int(KL));
        uint maxCol = min(N, int(gid) + int(KU) + 1);
        for(uint curCol = minCol; curCol < maxCol; curCol++)
        {
            uint AbIdx = order == RowMajor ? (gid * lda + (curCol - gid + KL)) : (curCol * lda + (gid - curCol + KU));
            result += A[AbIdx] * x[curCol * incx];
        }
    }
    else
    {
        uint minRow = max(0, int(gid) - int(KU));
        uint maxRow = min(M, int(gid) + int(KL) + 1);
        for(uint curRow = minRow; curRow < maxRow; curRow++)
        {
            uint AbIdx = order == RowMajor ? curRow * lda + gid - curRow + KL : (curRow - gid + KU)  + (gid) * lda;
            result += A[AbIdx] * x[curRow * incx];
        }
    }

    y[gid * incy] = alpha * result + beta * y[gid * incy];
}

template [[host_name("metalHgbmv")]]
kernel void metalGbmvWork<half>(const device OrderType& order [[buffer(0)]],
                                const device TransposeType& trans [[buffer(1)]],
                                const device int& M [[buffer(2)]],
                                const device int& N [[buffer(3)]],
                                const device int& KL [[buffer(4)]],
                                const device int& KU [[buffer(5)]],
                                const device half& alpha [[buffer(6)]],
                                const device half* A [[buffer(7)]],
                                const device int& lda [[buffer(8)]],
                                const device half* x [[buffer(9)]],
                                const device int& incx [[buffer(10)]],
                                const device half& beta [[buffer(11)]],
                                device half* y [[buffer(12)]],
                                const device int& incy [[buffer(13)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSgbmv")]]
kernel void metalGbmvWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device TransposeType& trans [[buffer(1)]],
                                 const device int& M [[buffer(2)]],
                                 const device int& N [[buffer(3)]],
                                 const device int& KL [[buffer(4)]],
                                 const device int& KU [[buffer(5)]],
                                 const device float& alpha [[buffer(6)]],
                                 const device float* A [[buffer(7)]],
                                 const device int& lda [[buffer(8)]],
                                 const device float* x [[buffer(9)]],
                                 const device int& incx [[buffer(10)]],
                                 const device float& beta [[buffer(11)]],
                                 device float* y [[buffer(12)]],
                                 const device int& incy [[buffer(13)]],
                                 uint gid [[thread_position_in_grid]]);
