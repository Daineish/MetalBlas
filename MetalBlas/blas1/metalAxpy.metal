//
//  metalAxpy.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void metalAxpyWork(const device int& N [[buffer(0)]],
                          const device T& alpha [[buffer(1)]],
                          const device T* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          device T* y [[buffer(4)]],
                          const device int& incy [[buffer(5)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    y[gid * incy] += alpha * x[gid * incx];
}

template [[host_name("metalHaxpy")]]
kernel void metalAxpyWork<half>(const device int& N [[buffer(0)]],
                                const device half& alpha [[buffer(1)]],
                                const device half* x [[buffer(2)]],
                                const device int& incx [[buffer(3)]],
                                device half* y [[buffer(4)]],
                                const device int& incy [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSaxpy")]]
kernel void metalAxpyWork<float>(const device int& N [[buffer(0)]],
                                const device float& alpha [[buffer(1)]],
                                const device float* x [[buffer(2)]],
                                const device int& incx [[buffer(3)]],
                                device float* y [[buffer(4)]],
                                const device int& incy [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]);
