//
//  referenceSwap.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

import Accelerate

func testSwap(_ N: __LAPACK_int, _ X: inout [Float], _ incx: __LAPACK_int, _ Y: inout [Float], _ incy: __LAPACK_int)
{
    cblas_sswap(N, &X, incx, &Y, incy)
}

func testSwap(_ N: __LAPACK_int, _ X: inout [Double], _ incx: __LAPACK_int, _ Y: inout [Double], _ incy: __LAPACK_int)
{
    cblas_dswap(N, &X, incx, &Y, incy)
}

public func refHswap(_ N: Int, _ X: inout [ Float16 ], _ incx: Int, _ Y: inout [ Float16 ], _ incy: Int)
{
    for i in 0...N - 1
    {
        let tmp = Y[i * incy];
        Y[i * incy] = X[i * incx];
        X[i * incx] = tmp;
    }
}

public func refSswap(_ N: Int, _ X: inout [ Float ], _ incx: Int, _ Y: inout [ Float ], _ incy: Int)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testSwap(N_la, &X, incx_la, &Y, incy_la)
}

public func refDswap(_ N: Int, _ X: inout [ Double ], _ incx: Int, _ Y: inout [ Double ], _ incy: Int)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testSwap(N_la, &X, incx_la, &Y, incy_la)
}
