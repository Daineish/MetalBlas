//
//  referenceDot.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

import Accelerate

func testSdot(_ N: __LAPACK_int, _ X: [Float], _ incx: __LAPACK_int, _ Y: [Float], _ incy: __LAPACK_int) -> Float
{
    return cblas_sdot(N, X, incx, Y, incy)
}

func testDdot(_ N: __LAPACK_int, _ X: [Double], _ incx: __LAPACK_int, _ Y: [Double], _ incy: __LAPACK_int) -> Double
{
    return cblas_ddot(N, X, incx, Y, incy)
}

public func refHdot(_ N: Int, _ X: [Float16], _ incx: Int, _ Y: [Float16], _ incy: Int) -> Float16
{
    // Float accumulate for now
    var sum : Float = 0
    for i in 0...N - 1
    {
        sum += Float(X[i * incx] * Y[i * incy])
    }

    return Float16(sum)
}

public func refSdot(_ N: Int, _ X: [ Float ], _ incx: Int, _ Y: [ Float ], _ incy: Int) -> Float
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    return testSdot(N_la, X, incx_la, Y, incy_la)
}

public func refDdot(_ N: Int, _ X: [ Double ], _ incx: Int, _ Y: [ Double ], _ incy: Int) -> Double
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    return testDdot(N_la, X, incx_la, Y, incy_la)
}

