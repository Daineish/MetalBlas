//
//  metalRot.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-26.
//

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void metalRotWork(const device int& N [[buffer(0)]],
                         device T* x [[buffer(1)]],
                         const device int& incx [[buffer(2)]],
                         device T* y [[buffer(3)]],
                         const device int& incy [[buffer(4)]],
                         const device T& c [[buffer(5)]],
                         const device T& s [[buffer(6)]],
                         uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    T tmp = c * x[gid * incx] + s * y[gid * incy];
    y[gid * incy] = c * y[gid * incy] - s * x[gid * incx];
    x[gid * incx] = tmp;
}

template [[host_name("metalHrot")]]
kernel void metalRotWork<half>(const device int& N [[buffer(0)]],
                               device half* x [[buffer(1)]],
                               const device int& incx [[buffer(2)]],
                               device half* y [[buffer(3)]],
                               const device int& incy [[buffer(4)]],
                               const device half& c [[buffer(5)]],
                               const device half& s [[buffer(6)]],
                               uint gid [[thread_position_in_grid]]);

template [[host_name("metalSrot")]]
kernel void metalRotWork<float>(const device int& N [[buffer(0)]],
                                device float* x [[buffer(1)]],
                                const device int& incx [[buffer(2)]],
                                device float* y [[buffer(3)]],
                                const device int& incy [[buffer(4)]],
                                const device float& c [[buffer(5)]],
                                const device float& s [[buffer(6)]],
                                uint gid [[thread_position_in_grid]]);
