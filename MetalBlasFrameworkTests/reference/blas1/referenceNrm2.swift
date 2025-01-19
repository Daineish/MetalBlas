//
//  referenceNrm2.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//


import Accelerate

func testSnrm2(_ N: __LAPACK_int, _ X: [Float], _ incx: __LAPACK_int) -> Float
{
    return cblas_snrm2(N, X, incx)
}

func testDnrm2(_ N: __LAPACK_int, _ X: [Double], _ incx: __LAPACK_int) -> Double
{
    return cblas_dnrm2(N, X, incx)
}

public func refHnrm2(_ N: Int, _ X: [ Float16 ], _ incx: Int) -> Float16
{
    // Float accumulate for now
    var sum : Float = 0
    for i in 0...N - 1
    {
        sum += Float(X[i] * X[i])
    }

    return Float16(sqrt(sum))
}

public func refSnrm2(_ N: Int, _ X: [ Float ], _ incx: Int) -> Float
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    return testSnrm2(N_la, X, incx_la)
}

public func refDnrm2(_ N: Int, _ X: [ Double ], _ incx: Int) -> Double
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    return testDnrm2(N_la, X, incx_la)
}
