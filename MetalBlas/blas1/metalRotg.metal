//
//  metalRotg.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-29.
//

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void metalRotgWork(device T* a [[buffer(0)]],
                          device T* b [[buffer(1)]],
                          device T* c [[buffer(2)]],
                          device T* s [[buffer(3)]],
                         uint gid [[thread_position_in_grid]])
{
    if(gid != 0)
        return;

    T scale = fabs(*a) + fabs(*b);
    if(scale == 0)
    {
        *c = 1.0;
        *a = *b = *s = 0.0;
    }
    else
    {
        T roe = fabs(*a) > fabs(*b) ? *a : *b;

        T tmpA = *a / scale;
        T tmpB = *b / scale;
        T r = scale * sqrt(tmpA * tmpA + tmpB * tmpB);
        r = copysign(r, roe);

        *c = *a / r;
        *s = *b / r;
        T z = 1.0;
        if(fabs(*a) > fabs(*b))
            z = *s;

        if(fabs(*b) >= fabs(*a) && *c != 0.0)
            z = 1.0 / *c;

        *a = r;
        *b = z;
    }
}

template [[host_name("metalHrotg")]]
kernel void metalRotgWork<half>(device half* a [[buffer(0)]],
                                device half* b [[buffer(1)]],
                                device half* c [[buffer(2)]],
                                device half* s [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSrotg")]]
kernel void metalRotgWork<float>(device float* a [[buffer(0)]],
                                 device float* b [[buffer(1)]],
                                 device float* c [[buffer(2)]],
                                 device float* s [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]);
