//
//  metalGemm.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

#include <metal_stdlib>
using namespace metal;

template <typename Ti, typename To, typename Ts, typename Tex>
kernel void metalGemmWork(const device int& M [[buffer(0)]],
                          const device int& N [[buffer(1)]],
                          const device int& K [[buffer(2)]],
                          const device Ts& alpha [[buffer(3)]],
                          const device Ti* A [[buffer(4)]],
                          const device int& lda [[buffer(5)]],
                          const device Ti* B [[buffer(6)]],
                          const device int& ldb [[buffer(7)]],
                          const device Ts& beta [[buffer(8)]],
                          device To* C [[buffer(9)]],
                          const device int& ldc [[buffer(10)]],
                          uint2 gid [[thread_position_in_grid]])
{
    uint globalRow = gid.x;
    uint globalCol = gid.y;
    if(globalRow >= uint(M) || globalCol >= uint(N))
        return;
    Tex acc = 0.0;
    for(int i = 0; i < K; i++)
    {
        acc += A[i * lda + globalRow] * B[globalCol * ldb + i];
    }
    C[globalCol * ldc + globalRow] = alpha * acc + beta * C[globalCol * ldc + globalRow];
}

template [[host_name("metalSgemm")]]
kernel void metalGemmWork<float, float, float, float>(const device int& M [[buffer(0)]],
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
                                               uint2 gid [[thread_position_in_grid]]);

// TODO: compute type?
template [[host_name("metalHgemm")]]
kernel void metalGemmWork<half, half, half, float>(const device int& M [[buffer(0)]],
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
                                               uint2 gid [[thread_position_in_grid]]);
