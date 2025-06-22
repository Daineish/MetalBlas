//
//  metalTpmv.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-06-21.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <typename T>
kernel void metalTpmvWork(const device OrderType& order [[buffer(0)]],
                          const device UploType& uplo [[buffer(1)]],
                          const device TransposeType& trans [[buffer(2)]],
                          const device DiagType& diag [[buffer(3)]],
                          const device int& N [[buffer(4)]],
                          const device T* Ap [[buffer(5)]],
                          device T* x [[buffer(6)]],
                          const device int& incx [[buffer(7)]],
                          device T* work [[buffer(8)]],
                          uint gid [[thread_position_in_grid]])
{
    bool accessUpper = (uplo == FillUpper && trans == NoTranspose) || (uplo == FillLower && trans != NoTranspose);
    if(N == 0 || gid >= uint(N))
        return;

    T sum = 0;
    const int row = gid;

    if(accessUpper)
    {
        for(int col = row; col < N; col++)
        {
            if(trans == NoTranspose)
            {
                int ApIdx = order == RowMajor ? (row * N + col - row) - ((row - 1) * row) / 2 : (col * (col + 1) / 2 + row);
                if(row < N && row >= 0 && col < N && col >= 0)
                    sum += (col == row && diag == Unit) ? work[col] : Ap[ApIdx] * work[col];
            }
            else
            {
                int ApIdx = order == RowMajor ? (col * (col + 1) / 2 + row) : (row * N + col - row) - ((row - 1) * row) / 2;
                sum += (col == row && diag == Unit) ? work[col] : Ap[ApIdx] * work[col];
            }
        }
    }
    else
    {
        for(int col = 0; col <= row; col++)
        {
            if(trans == NoTranspose)
            {
                int ApIdx = order == RowMajor ? (row * (row + 1) / 2 + col) : (col * N + row - col) - ((col - 1) * col) / 2;
                sum += (col == row && diag == Unit) ? work[col] : Ap[ApIdx] * work[col];
            }
            else
            {
                int ApIdx = order == RowMajor ? (col * N + row - col) - ((col - 1) * col) / 2 : (row * (row + 1) / 2 + col);
                sum += (col == row && diag == Unit) ? work[col] : Ap[ApIdx] * work[col];
            }
        }
    }

    if(row < N && row >= 0)
        x[row * incx] = sum;
}

template [[host_name("metalHtpmv")]]
kernel void metalTpmvWork<half>(const device OrderType& order [[buffer(0)]],
                                const device UploType& uplot [[buffer(1)]],
                                const device TransposeType& trans [[buffer(2)]],
                                const device DiagType& diag [[buffer(3)]],
                                const device int& N [[buffer(4)]],
                                const device half* A [[buffer(5)]],
                                device half* x [[buffer(6)]],
                                const device int& incx [[buffer(7)]],
                                device half* work [[buffer(8)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalStpmv")]]
kernel void metalTpmvWork<float>(const device OrderType& order [[buffer(0)]],
                                 const device UploType& uplo [[buffer(1)]],
                                 const device TransposeType& trans [[buffer(2)]],
                                 const device DiagType& diag [[buffer(3)]],
                                 const device int& N [[buffer(4)]],
                                 const device float* A [[buffer(5)]],
                                 device float* x [[buffer(6)]],
                                 const device int& incx [[buffer(7)]],
                                 device float* work [[buffer(8)]],
                                 uint gid [[thread_position_in_grid]]);
