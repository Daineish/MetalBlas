//
//  metalCopy.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void metalCopyWork(const device int& N [[buffer(0)]],
                          const device T* x[[buffer(1)]],
                          const device int& incx[[buffer(2)]],
                          device T* y[[buffer(3)]],
                          const device int& incy[[buffer(4)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    y[gid * incy] = x[gid * incx];
}
 
template [[host_name("metalHcopy")]]
kernel void metalCopyWork<half>(const device int& N [[buffer(0)]],
                          const device half* x[[buffer(1)]],
                          const device int& incx[[buffer(2)]],
                          device half* y[[buffer(3)]],
                          const device int& incy[[buffer(4)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalScopy")]]
kernel void metalCopyWork<float>(const device int& N [[buffer(0)]],
                          const device float* x[[buffer(1)]],
                          const device int& incx[[buffer(2)]],
                          device float* y[[buffer(3)]],
                          const device int& incy[[buffer(4)]],
                                uint gid [[thread_position_in_grid]]);
