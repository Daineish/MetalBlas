//
//  metalRotm.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-26.
//

#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void metalRotmWork(const device int& N [[buffer(0)]],
                         device T* x [[buffer(1)]],
                         const device int& incx [[buffer(2)]],
                         device T* y [[buffer(3)]],
                         const device int& incy [[buffer(4)]],
                         const device T* P [[buffer(5)]],
                         uint gid [[thread_position_in_grid]])
{
    if(gid >= uint(N))
        return;

    // fine with fp comparison since these are exact representations
    T flag = P[0];
    if(flag == -2.0)
        return;

    // Error?
    bool flagn1 = (flag == T(-1.0));
    bool flagz = (flag == T(0.0));
    bool flagp1 = (flag == T(1.0));
    if(!flagn1 && !flagz && !flagp1)
        return;

    T sh11 = (flagn1 || flagp1) ? P[1] : T(1.0);
    T sh21 = (flagn1 || flagz) ? P[2] : T(-1.0);
    T sh12 = (flagn1 || flagz) ? P[3] : T(1.0);
    T sh22 = (flagn1 || flagp1) ? P[4] : T(1.0);

    T tmp = x[gid * incx];
    x[gid * incx] = float(tmp * sh11) + float(y[gid * incy] * sh12);
    y[gid * incy] = float(tmp * sh21) + float(y[gid * incy] * sh22);
}

template [[host_name("metalHrotm")]]
kernel void metalRotmWork<half>(const device int& N [[buffer(0)]],
                               device half* x [[buffer(1)]],
                               const device int& incx [[buffer(2)]],
                               device half* y [[buffer(3)]],
                               const device int& incy [[buffer(4)]],
                               const device half* P [[buffer(5)]],
                               uint gid [[thread_position_in_grid]]);

template [[host_name("metalSrotm")]]
kernel void metalRotmWork<float>(const device int& N [[buffer(0)]],
                                device float* x [[buffer(1)]],
                                const device int& incx [[buffer(2)]],
                                device float* y [[buffer(3)]],
                                const device int& incy [[buffer(4)]],
                                 const device float* P [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]);
