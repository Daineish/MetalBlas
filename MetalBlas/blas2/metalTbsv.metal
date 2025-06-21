//
//  metalTbsv.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-14.
//

#include <metal_stdlib>
#include "../include/metalEnums.metal"
using namespace metal;

template <int BLOCK, typename T>
kernel void metalTbsvWork(const device OrderType& order [[buffer(0)]],
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
                          uint tid [[thread_position_in_threadgroup]],
                          uint gid [[thread_position_in_grid]])
{
    if(N == 0)
        return;

    bool accessLower = (uplo == FillLower && trans == NoTranspose) || (uplo == FillUpper && trans != NoTranspose);

    threadgroup T s_x[BLOCK];
    // TODO: shared A? not super straightforward

    int bStart = accessLower ? 0 : N - BLOCK;
    int bInc = accessLower ? BLOCK : -BLOCK;
    for(int b = bStart; b < N && b > -BLOCK; b += bInc)
    {
        bool inBounds = tid + b >= 0 && tid + b < uint(N);
        if(inBounds)
            s_x[tid] = x[(tid + b) * incx];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int blockStart = accessLower ? 0 : BLOCK - 1;
        int blockInc = accessLower ? 1 : -1;
        for(int blockCol = blockStart; blockCol < BLOCK && blockCol >= 0; blockCol += blockInc)
        {
            if(inBounds && diag == NonUnit && tid == uint(blockCol))
            {
                // solve
                int aRow = tid + b;
                int aCol = aRow;
                int AbIdx = accessLower ? (trans == NoTranspose ? (order == RowMajor ? (aRow * lda + (aCol - aRow + K)) : (aCol * lda + (aRow - aCol)))
                                                                : (order == RowMajor ? (aCol * lda + (aRow - aCol)) : (aRow * lda + (aCol - aRow + K))))
                                        : (trans == NoTranspose ? (order == RowMajor ? (aRow * lda + aCol - aRow) : (aCol * lda + aRow - aCol + K))
                                                                : (order == RowMajor ? (aCol * lda + aRow - aCol + K) : (aRow * lda + aCol - aRow)));
                s_x[tid] = s_x[tid] / A[AbIdx];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(inBounds && ((accessLower && tid > uint(blockCol)) || (!accessLower && tid < uint(blockCol))))
            {
                int aRow = tid + b;
                int aCol = b + blockCol;
                if(aCol < N && aCol >= 0 && ((accessLower && aCol <= aRow && aCol >= aRow - K) || ((!accessLower && aCol >= aRow && aCol <= aRow + K))))
                {
                    int AbIdx = accessLower ? (trans == NoTranspose ? (order == RowMajor ? (aRow * lda + (aCol - aRow + K)) : (aCol * lda + (aRow - aCol)))
                                                                    : (order == RowMajor ? (aCol * lda + (aRow - aCol)) : (aRow * lda + (aCol - aRow + K))))
                                            : (trans == NoTranspose ? (order == RowMajor ? (aRow * lda + aCol - aRow) : (aCol * lda + aRow - aCol + K))
                                                                    : (order == RowMajor ? (aCol * lda + aRow - aCol + K) : (aRow * lda + aCol - aRow)));
                    s_x[tid] -= A[AbIdx] * s_x[blockCol];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        int blockRowStart = accessLower ? b + BLOCK : b - BLOCK;
        int blockRowInc = accessLower ? BLOCK : -BLOCK;
        for(int blockRow = blockRowStart; blockRow < N && blockRow >= 0; blockRow += blockRowInc)
        {
            T sum = 0;
            for(int blockCol = 0; blockCol < BLOCK; blockCol++)
            {
                int aRow = blockRow + tid;
                int aCol = b + blockCol;
                int AbIdx = accessLower ? (trans == NoTranspose ? (order == RowMajor ? (aRow * lda + (aCol - aRow + K)) : (aCol * lda + (aRow - aCol)))
                                                                : (order == RowMajor ? (aCol * lda + (aRow - aCol)) : (aRow * lda + (aCol - aRow + K))))
                                        : (trans == NoTranspose ? (order == RowMajor ? (aRow * lda + aCol - aRow) : (aCol * lda + aRow - aCol + K))
                                                                : (order == RowMajor ? (aCol * lda + aRow - aCol + K) : (aRow * lda + aCol - aRow)));
                
                if(aCol < N && aRow < N && aCol >= 0 && aRow >= 0)
                {
                    if(aCol == aRow && diag == Unit)
                        sum += s_x[blockCol];
                    else if((accessLower && aCol >= aRow - K && aCol < aRow) || (!accessLower && aCol > aRow && aCol <= aRow + K))
                        sum += A[AbIdx] * s_x[blockCol];
                }
            }

            x[(tid + blockRow) * incx] -= sum;
        }
        
        if(inBounds)
        {
            x[(tid + b) * incx] = s_x[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("metalHtbsv")]]
kernel void metalTbsvWork<TBSV_BLOCK, half>(const device OrderType& order [[buffer(0)]],
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
                                uint tid [[thread_position_in_threadgroup]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalStbsv")]]
kernel void metalTbsvWork<TBSV_BLOCK, float>(const device OrderType& order [[buffer(0)]],
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
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint gid [[thread_position_in_grid]]);
