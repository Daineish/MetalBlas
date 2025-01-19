//
//  referenceCopy.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

import Accelerate

func testCopy(_ N: __LAPACK_int, _ X: [Float], _ incx: __LAPACK_int, _ Y: inout [Float], _ incy: __LAPACK_int)
{
    cblas_scopy(N, X, incx, &Y, incy)
}

func testCopy(_ N: __LAPACK_int, _ X: [Double], _ incx: __LAPACK_int, _ Y: inout [Double], _ incy: __LAPACK_int)
{
    cblas_dcopy(N, X, incx, &Y, incy)
}

public func refHcopy(_ N: Int, _ X: [ Float16 ], _ incx: Int, _ Y: inout [ Float16 ], _ incy: Int)
{
    for i in 0...N - 1
    {
        Y[i * incy] = X[i * incx];
    }
}

public func refScopy(_ N: Int, _ X: [ Float ], _ incx: Int, _ Y: inout [ Float ], _ incy: Int)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testCopy(N_la, X, incx_la, &Y, incy_la)
}

public func refDcopy(_ N: Int, _ X: [ Double ], _ incx: Int, _ Y: inout [ Double ], _ incy: Int)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testCopy(N_la, X, incx_la, &Y, incy_la)
}
