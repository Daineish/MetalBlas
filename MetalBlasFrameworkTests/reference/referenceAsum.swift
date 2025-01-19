//
//  referenceAsum.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-17.
//

import Accelerate

func testSasum(_ N: __LAPACK_int, _ X: [Float], _ incx: __LAPACK_int) -> Float
{
    return cblas_sasum(N, X, incx)
}

func testDasum(_ N: __LAPACK_int, _ X: [Double], _ incx: __LAPACK_int) -> Double
{
    return cblas_dasum(N, X, incx)
}

public func refHasum(_ N: Int, _ X: [ Float16 ], _ incx: Int) -> Float16
{
    // Float accumulate for now
    var sum : Float = 0
    for i in 0...N - 1
    {
        sum += abs(Float(X[i]))
    }

    return Float16(sum)
}

public func refSasum(_ N: Int, _ X: [ Float ], _ incx: Int) -> Float
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    return testSasum(N_la, X, incx_la)
}

public func refDasum(_ N: Int, _ X: [ Double ], _ incx: Int) -> Double
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    return testDasum(N_la, X, incx_la)
}

