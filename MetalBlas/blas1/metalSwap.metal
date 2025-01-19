//
//  metalSwap.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void metalSwapWork(const device int& N [[buffer(0)]],
                          device T* x[[buffer(1)]],
                          const device int& incx[[buffer(2)]],
                          device T* y[[buffer(3)]],
                          const device int& incy[[buffer(4)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    T tmp = y[gid * incy];
    y[gid * incy] = x[gid * incx];
    x[gid * incx] = tmp;
}
 
template [[host_name("metalHswap")]]
kernel void metalSwapWork<half>(const device int& N [[buffer(0)]],
                          device half* x[[buffer(1)]],
                          const device int& incx[[buffer(2)]],
                          device half* y[[buffer(3)]],
                          const device int& incy[[buffer(4)]],
                          uint gid [[thread_position_in_grid]]);

template [[host_name("metalSswap")]]
kernel void metalSwapWork<float>(const device int& N [[buffer(0)]],
                          device float* x[[buffer(1)]],
                          const device int& incx[[buffer(2)]],
                          device float* y[[buffer(3)]],
                          const device int& incy[[buffer(4)]],
                          uint gid [[thread_position_in_grid]]);
