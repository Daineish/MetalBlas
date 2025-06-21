//
//  metalTbmv.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalTbmvWork(const device OrderType& order [[buffer(0)]],
                          const device UploType& uplo [[buffer(1)]],
                          const device TransposeType& trans [[buffer(2)]],
                          const device DiagType& diag [[buffer(3)]],
                          const device int& N [[buffer(4)]],
                          const device int& K [[buffer(5)]],
                          const device T* A [[buffer(6)]],
                          const device int& lda [[buffer(7)]],
                          device T* x [[buffer(8)]],
                          const device int& incx [[buffer(9)]],
                          device T* work [[buffer(10)]],
                          uint gid [[thread_position_in_grid]])
{
    if(N == 0 || gid >= uint(N))
        return;

    bool accessUpper = (uplo == FillUpper && trans == NoTranspose) || (uplo == FillLower && trans != NoTranspose);

    T sum = 0;
    const int row = gid;

    if(accessUpper)
    {
        int minCol = row;
        int maxCol = N > row + K + 1 ? row + K + 1 : N;
        for(int col = minCol; col < maxCol; col++)
        {
            if(trans == NoTranspose)
            {
                int AbIdx = order == RowMajor ? (row * lda + (col - row)) : (col * lda + (row - col + K));
                if(row < N && row >= 0 && col < N && col >= 0)
                    sum += (col == row && diag == Unit) ? work[col] : A[AbIdx] * work[col];
            }
            else
            {
                int AbIdx = order == RowMajor ? (col * lda + (row - col + K)) : (row * lda + (col - row));
                sum += (col == row && diag == Unit) ? work[col] : A[AbIdx] * work[col];
            }
        }
    }
    else
    {
        int minCol = 0 > row - K ? 0 : row - K;
        int maxCol = row + 1;
        for(int col = minCol; col < maxCol; col++)
        {
            if(trans == NoTranspose)
            {
                int AbIdx = order == RowMajor ? (row * lda + (col - row + K)) : (col * lda + (row - col));
                sum += (col == row && diag == Unit) ? work[col] : A[AbIdx] * work[col];
            }
            else
            {
                int AbIdx = order == RowMajor ? (col * lda + (row - col)) : (row * lda + (col - row + K));
                sum += (col == row && diag == Unit) ? work[col] : A[AbIdx] * work[col];
            }
        }
    }

    if(row < N && row >= 0)
        x[row * incx] = sum;
}

template [[host_name("metalHtbmv")]]
kernel void metalTbmvWork<half>(const device OrderType& order [[buffer(0)]],
                                const device UploType& uplot [[buffer(1)]],
                                const device TransposeType& trans [[buffer(2)]],
                                const device DiagType& diag [[buffer(3)]],
                                const device int& N [[buffer(4)]],
                                const device int& K [[buffer(5)]],
                                const device half* A [[buffer(6)]],
                                const device int& lda [[buffer(7)]],
                                device half* x [[buffer(8)]],
                                const device int& incx [[buffer(9)]],
                                device half* work [[buffer(10)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalStbmv")]]
kernel void metalTbmvWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device UploType& uplo [[buffer(1)]],
                                 const device TransposeType& trans [[buffer(2)]],
                                 const device DiagType& diag [[buffer(3)]],
                                 const device int& N [[buffer(4)]],
                                 const device int& K [[buffer(5)]],
                                 const device float* A [[buffer(6)]],
                                 const device int& lda [[buffer(7)]],
                                 device float* x [[buffer(8)]],
                                 const device int& incx [[buffer(9)]],
                                 device float* work [[buffer(10)]],
                                 uint gid [[thread_position_in_grid]]);
