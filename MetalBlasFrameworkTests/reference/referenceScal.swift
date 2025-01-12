//
//  referenceScal.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Accelerate

func testSscal(_ N: __LAPACK_int, _ alpha: Float, _ X: inout [Float], _ incx: __LAPACK_int)
{
    cblas_sscal(N, alpha, &X, incx)
}

func testDscal(_ N: __LAPACK_int, _ alpha: Double, _ X: inout [Double], _ incx: __LAPACK_int)
{
    cblas_dscal(N, alpha, &X, incx)
}

public func refHscal(_ N: Int, _ alpha: Float16, _ X: inout [ Float16 ], _ incx: Int)
{
    for i in 0...N - 1
    {
        X[i * incx] *= alpha
    }
}

public func refSscal(_ N: Int, _ alpha: Float, _ X: inout [ Float ], _ incx: Int)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    testSscal(N_la, alpha, &X, incx_la)
}

public func refDscal(_ N: Int, _ alpha: Double, _ X: inout [ Double ], _ incx: Int)
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    testDscal(N_la, alpha, &X, incx_la)
}

