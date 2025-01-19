//
//  metalScal.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void metalScalWork(const device int& N [[buffer(0)]],
                          const device T& alpha [[buffer(1)]],
                          device T* x [[buffer(2)]],
                          const device int& incx [[buffer(3)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    x[gid * incx] *= alpha;
}

template [[host_name("metalSscal")]]
kernel void metalScalWork<float>(const device int& N [[buffer(0)]],
                                 const device float& alpha [[buffer(1)]],
                                 device float* x [[buffer(2)]],
                                 const device int& incx [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]);

template [[host_name("metalHscal")]]
kernel void metalScalWork<half>(const device int& N [[buffer(0)]],
                                 const device half& alpha [[buffer(1)]],
                                 device half* x [[buffer(2)]],
                                 const device int& incx [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]);
