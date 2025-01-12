//
//  MetalBlas.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-04.
//

#include <metal_stdlib>
using namespace metal;

kernel void metalSgemm(const device int& M [[buffer(0)]],
                          const device int& N [[buffer(1)]],
                          const device int& K [[buffer(2)]],
                          const device float& alpha [[buffer(3)]],
                          const device float* A [[buffer(4)]],
                          const device int& lda [[buffer(5)]],
                          const device float* B [[buffer(6)]],
                          const device int& ldb [[buffer(7)]],
                          const device float& beta [[buffer(8)]],
                          device float* C [[buffer(9)]],
                          const device int& ldc [[buffer(10)]],
                          uint2 gid [[thread_position_in_grid]])
{
    uint globalRow = gid.x;
    uint globalCol = gid.y;
    if(globalRow >= uint(M) || globalCol >= uint(N))
        return;
    float acc = 0.0f;
    for(int i = 0; i < K; i++)
    {
        acc += A[i * lda + globalRow] * B[globalCol * ldb + i];
    }
    C[globalCol * ldc + globalRow] = alpha * acc + beta * C[globalCol * ldc + globalRow];
}

kernel void metalHgemm(const device int& M [[buffer(0)]],
                          const device int& N [[buffer(1)]],
                          const device int& K [[buffer(2)]],
                          const device half& alpha [[buffer(3)]],
                          const device half* A [[buffer(4)]],
                          const device int& lda [[buffer(5)]],
                          const device half* B [[buffer(6)]],
                          const device int& ldb [[buffer(7)]],
                          const device half& beta [[buffer(8)]],
                          device half* C [[buffer(9)]],
                          const device int& ldc [[buffer(10)]],
                          uint2 gid [[thread_position_in_grid]])
{
    uint globalRow = gid.x;
    uint globalCol = gid.y;
    if(globalRow >= uint(M) || globalCol >= uint(N))
        return;
    float acc = 0.0f;
    for(int i = 0; i < K; i++)
    {
        acc += A[i * lda + globalRow] * B[globalCol * ldb + i];
    }
    C[globalCol * ldc + globalRow] = alpha * acc + beta * C[globalCol * ldc + globalRow];
}

kernel void metalSaxpy(const device int& N [[buffer(0)]],
                          const device float& alpha [[buffer(1)]],
                          const device float* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          device float* y [[buffer(4)]],
                          const device int& incy [[buffer(5)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    y[gid * incy] += alpha * x[gid * incx];
}

kernel void metalHaxpy(const device int& N [[buffer(0)]],
                          const device half& alpha [[buffer(1)]],
                          const device half* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          device half* y [[buffer(4)]],
                          const device int& incy [[buffer(5)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    y[gid * incy] += alpha * x[gid * incx];
}

kernel void metalHscal(const device int& N [[buffer(0)]],
                          const device half& alpha [[buffer(1)]],
                          device half* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    x[gid * incx] *= alpha;
}

kernel void metalSscal(const device int& N [[buffer(0)]],
                          const device float& alpha [[buffer(1)]],
                          device float* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    x[gid * incx] *= alpha;
}

