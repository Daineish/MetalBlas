//
//  referenceAmin.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

// Keeping same pattern here even though can't call Accelerate/cblas
public func refAmin<T: BinaryFloatingPoint>(_ N: Int, _ X: [T], _ incx: Int) -> Int
{
    if N < 0
    {
        return -1
    }

    var minVal = abs(X[0])
    var minIdx = 0
    for i in 1...N - 1
    {
        if abs(X[i]) < minVal
        {
            minVal = abs(X[i])
            minIdx = i
        }
    }

    return minIdx
}

public func refIhamin(_ N: Int, _ X: [ Float16 ], _ incx: Int) -> Int
{
    return refAmin(N, X, incx)
}

public func refIsamin(_ N: Int, _ X: [ Float ], _ incx: Int) -> Int
{
    return refAmin(N, X, incx)
}

public func refIdamin(_ N: Int, _ X: [ Double ], _ incx: Int) -> Int
{
    return refAmin(N, X, incx)
}

