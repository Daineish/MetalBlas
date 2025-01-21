//
//  referenceAmax.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

import Accelerate

func testIsamax(_ N: __LAPACK_int, _ X: [Float], _ incx: __LAPACK_int) -> Int
{
    return cblas_isamax(N, X, incx)
}

func testIdamax(_ N: __LAPACK_int, _ X: [Double], _ incx: __LAPACK_int) -> Int
{
    return cblas_idamax(N, X, incx)
}

public func refIhamax(_ N: Int, _ X: [ Float16 ], _ incx: Int) -> Int
{
    if N < 0
    {
        return -1
    }

    var maxVal = abs(X[0])
    var maxIdx = 0
    for i in 1...N - 1
    {
        if abs(X[i]) > maxVal
        {
            maxVal = abs(X[i])
            maxIdx = i
        }
    }

    return maxIdx
}

public func refIsamax(_ N: Int, _ X: [ Float ], _ incx: Int) -> Int
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    return testIsamax(N_la, X, incx_la)
}

public func refIdamax(_ N: Int, _ X: [ Double ], _ incx: Int) -> Int
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    return testIdamax(N_la, X, incx_la)
}

