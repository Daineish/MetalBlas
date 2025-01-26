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

public func printIfNotEqual<T: Numeric>(_ outMetal: [ T ], _ outRef: [ T ]) -> Bool
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

public func printIfNotEqual<T: Numeric>(_ outMetal: T, _ outRef: T) -> Bool
{
    if outMetal != outRef
    {
        print("Error: ref = ", outRef, "; got out = ", outMetal)
        return false
    }
    return true
}

public func printIfNotNear<T: BinaryFloatingPoint>(_ outMetal: [ T ], _ outRef: [ T ], _ N: Int) -> Bool
{
    if outMetal.count != outRef.count
    {
        print("Error: Trying to compare arrays of different sizes")
        return false
    }

    for i in 0...outMetal.count - 1
    {
        let outRefi = outRef[i]
        let outMeti = outMetal[i]
        let epsilon = 2 * (outRefi.nextUp - outRefi)
        let lowerBound = outRefi - epsilon * sqrt(T(N))
        let upperBound = outRefi + epsilon * sqrt(T(N))
        let rg : ClosedRange = lowerBound...upperBound
        
        if !(rg ~= outMeti)
        {
            print("Error: ref[", i, " ] = ", outRefi, "; got out[", i, "] = ", outMeti)
            return false
        }
    }

    return true
}

public func printIfNotNear<T: BinaryFloatingPoint>(_ outMetal: T, _ outRef: T, _ N : Int) -> Bool
{
    let epsilon = 2 * (outRef.nextUp - outRef)
    let lowerBound = outRef - epsilon * sqrt(T(N))
    let upperBound = outRef + epsilon * sqrt(T(N))
    let rg : ClosedRange = lowerBound...upperBound
    
    if !(rg ~= outMetal)
    {
        print("Error: ref = ", outRef, "; got out = ", outMetal)
        return false
    }
    return true
}

public func initRandom(_ arr: inout [Double], _ size: Int, _ range: ClosedRange<Double> = (0...10))
{
    for _ in 0...size - 1
    {
        let neg = Bool.random()
        let val = Double.random(in: range)
        arr.append(neg ? -val : val)
    }
}

public func initRandom(_ arr: inout [Float], _ size: Int, _ range: ClosedRange<Float> = (0...10))
{
    for _ in 0...size - 1
    {
        let neg = Bool.random()
        let val = Float.random(in: range)
        arr.append(neg ? -val : val)
    }
}

public func initRandom(_ arr: inout [Float16], _ size: Int, _ range: ClosedRange<Float16> = (0...10))
{
    for _ in 0...size - 1
    {
        let neg = Bool.random()
        let val = Float16.random(in: range)
        arr.append(neg ? -val : val)
    }
}

public func initRandom(_ arr: inout [Int], _ size: Int, _ range: ClosedRange<Int> = (0...10))
{
    for _ in 0...size - 1
    {
        let neg = Bool.random()
        let val = Int.random(in: range)
        arr.append(neg ? -val : val)
    }
}
