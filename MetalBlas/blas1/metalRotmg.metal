//
//  metalRotmg.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

#include <metal_stdlib>
using namespace metal;

template <typename T, int GAMMA>
kernel void metalRotmgWork(device T* d1 [[buffer(0)]],
                          device T* d2 [[buffer(1)]],
                          device T* b1 [[buffer(2)]],
                          const device T* b2 [[buffer(3)]],
                          device T* P [[buffer(4)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid != 0)
        return;

    T gam = GAMMA;
    T gamsq = gam * gam;
    T rgamsq = 1.0 / gamsq;

    T sflag = -1;
    T sh11 = 0, sh12 = 0, sh21 = 0, sh22 = 0;

    if(*d1 < 0)
    {
        *d1 = 0, *d2 = 0, *b1 = 0;
    }
    else
    {
        T p2 = *d2 * *b2;
        if(p2 == 0)
        {
            P[0] = -2;
            return;
        }

        T p1 = *d1 * *b1;
        T q2 = p2 * *b2;
        T q1 = p1 * *b1;
        T su = 0;

        if(abs(q1) > abs(q2))
        {
            sh21 = -(*b2) / *b1;
            sh12 = p2 / p1;

            T su = 1 - sh12 * sh21;
            if(su > 0)
            {
                sflag = 0;
                *d1 = *d1 / su;
                *d2 = *d2 / su;
                *b1 = *b1 * su;
            }
        }
        else
        {
            if(q2 < 0)
            {
                sflag = -1;
                sh11 = 0, sh12 = 0, sh21 = 0, sh22 = 0;
                *d1 = 0, *d2 = 0, *b1 = 0;
            }
            else
            {
                sflag = 1;
                sh11 = p1 / p2;
                sh22 = *b1 / *b2;
                su = 1.0 + sh11 * sh22;
                T tmp = *d2 / su;
                *d2 = *d1 / su;
                *d1 = tmp;
                *b1 = *b2 * su;
            }
        }

        if(*d1 != 0)
        {
            while(*d1 <= rgamsq || *d1 >= gamsq)
            {
                if(sflag == 0)
                    sh11 = 1, sh22 = 1, sflag = -1;
                else
                    sh21 = -1, sh12 = 1, sflag = -1;

                if(*d1 <= rgamsq)
                {
                    *d1 = *d1 * gamsq;
                    *b1 = *b1 / gam;
                    sh11 = sh11 / gam;
                    sh12 = sh12 / gam;
                }
                else
                {
                    *d1 = *d1 / gamsq;
                    *b1 = *b1 * gam;
                    sh11 = sh11 * gam;
                    sh12 = sh12 * gam;
                }
            }
        }
    }

    if(sflag < 0)
    {
        P[1] = sh11; P[2] = sh21;
        P[3] = sh12; P[4] = sh22;
    }
    else if(sflag == 0)
    {
        P[2] = sh21; P[3] = sh12;
    }
    else
    {
        P[1] = sh11; P[4] = sh22;
    }

    P[0] = sflag;
}

template [[host_name("metalHrotmg")]]
kernel void metalRotmgWork<half, 128>(device half* d1 [[buffer(0)]],
                                device half* d2 [[buffer(1)]],
                                device half* b1 [[buffer(2)]],
                                const device half* b2 [[buffer(3)]],
                                device half* P [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]);

template [[host_name("metalSrotmg")]]
kernel void metalRotmgWork<float, 4096>(device float* d1 [[buffer(0)]],
                                 device float* d2 [[buffer(1)]],
                                 device float* b1 [[buffer(2)]],
                                 const device float* b2 [[buffer(3)]],
                                 device float* P [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]);
