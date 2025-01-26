//
//  referenceRot.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-25.
//

import Accelerate

func testRot(_ N: __LAPACK_int, _ X: inout [Float], _ incx: __LAPACK_int, _ Y: inout [Float], _ incy: __LAPACK_int, _ c: Float, _ s: Float)
{
    cblas_srot(N, &X, incx, &Y, incy, c, s)
}

func testRot(_ N: __LAPACK_int, _ X: inout [Double], _ incx: __LAPACK_int, _ Y: inout [Double], _ incy: __LAPACK_int, _ c: Double, _ s: Double)
{
    cblas_drot(N, &X, incx, &Y, incy, c, s)
}

public func refHrot(_ N: Int, _ X: inout [ Float16 ], _ incx: Int, _ Y: inout [ Float16 ], _ incy: Int, _ c: Float16, _ s: Float16)
{
    for i in 0...N - 1
    {
        let tmp = c * X[i * incx] + s * Y[i * incy]
        Y[i * incy] = c * Y[i * incy] - s * X[i * incx]
        X[i * incx] = tmp
    }
}

public func refSrot(_ N: Int, _ X: inout [ Float ], _ incx: Int, _ Y: inout [ Float ], _ incy: Int, _ c: Float, _ s: Float)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testRot(N_la, &X, incx_la, &Y, incy_la, c, s)
}

public func refDrot(_ N: Int, _ X: inout [ Double ], _ incx: Int, _ Y: inout [ Double ], _ incy: Int, _ c: Double, _ s: Double)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testRot(N_la, &X, incx_la, &Y, incy_la, c, s)
}
