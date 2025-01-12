//
//  helpers.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Accelerate

public func cblasTrans(transA: TransposeType) -> CBLAS_TRANSPOSE
{
    if transA == .Transpose
    {
        return CblasTrans
    }
    else if transA == .NoTranspose
    {
        return CblasNoTrans
    }
    return CblasConjTrans
}

public func printIfNotEqual<T: BinaryFloatingPoint>(_ outMetal: [ T ], _ outRef: [ T ]) -> Bool
{
    if outMetal.count != outRef.count
    {
        print("Error: Trying to compare arrays of different sizes")
        return false
    }

    for i in 0...outMetal.count - 1
    {
        if outMetal[i] != outRef[i]
        {
            print("Error: ref[", i, " ] = ", outRef[i], "; got out[", i, "] = ", outMetal[i])
            return false
        }
    }

    return true
}

public func initRandom<T: BinaryFloatingPoint>(_ arr: inout [ T ], _ size: Int)
{
    for _ in 0...size - 1
    {
         arr.append(T(Int.random(in: -10...10)))
    }
}
