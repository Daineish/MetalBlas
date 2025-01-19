//
//  referenceAxpy.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Accelerate

func testAxpy(_ N: __LAPACK_int, _ alpha: Float, _ X: [Float], _ incx: __LAPACK_int, _ Y: inout [Float], _ incy: __LAPACK_int)
{
    cblas_saxpy(N, alpha, X, incx, &Y, incy)
}

func testAxpy(_ N: __LAPACK_int, _ alpha: Double, _ X: [Double], _ incx: __LAPACK_int, _ Y: inout [Double], _ incy: __LAPACK_int)
{
    cblas_daxpy(N, alpha, X, incx, &Y, incy)
}

public func refHaxpy(_ N: Int, _ alpha: Float16, _ X: [ Float16 ], _ incx: Int, _ Y: inout [ Float16 ], _ incy: Int)
{
    for i in 0...N - 1
    {
        Y[i * incy] += alpha * X[i * incx]
    }
}

public func refSaxpy(_ N: Int, _ alpha: Float, _ X: [ Float ], _ incx: Int, _ Y: inout [ Float ], _ incy: Int)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testAxpy(N_la, alpha, X, incx_la, &Y, incy_la)
}

public func refDaxpy(_ N: Int, _ alpha: Double, _ X: [ Double ], _ incx: Int, _ Y: inout [ Double ], _ incy: Int)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testAxpy(N_la, alpha, X, incx_la, &Y, incy_la)
}
