//
//  helpers.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Accelerate

public enum InitType
{
    case FloatingPoint
    case Integer
    case DiagDominant
}

public func cblasOrder(_ order: OrderType) -> CBLAS_ORDER
{
    if order == .RowMajor
    {
        return CblasRowMajor
    }
    return CblasColMajor
}

public func cblasTrans(_ transA: TransposeType) -> CBLAS_TRANSPOSE
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

public func cblasUplo(_ uplo: UploType) -> CBLAS_UPLO
{
    if uplo == .FillLower
    {
        return CblasLower
    }

    return CblasUpper
}

public func cblasDiag(_ diag: DiagType) -> CBLAS_DIAG
{
    if diag == .Unit
    {
        return CblasUnit
    }

    return CblasNonUnit
}

public func printIfNotEqual<T: Numeric>(_ outMetal: [ T ], _ outRef: [ T ], _ pr: Bool = true, _ prNote: String = "") -> Bool
{
    if outMetal.count != outRef.count
    {
        print("Error: Trying to compare arrays of different sizes, our size: ", outMetal.count, " ref size: ", outRef.count)
        return false
    }

    for i in 0...outMetal.count - 1
    {
        if outMetal[i] != outRef[i]
        {
            if pr
            {
                print(prNote, "Error: ref[", i, "] = ", outRef[i], "; got out[", i, "] = ", outMetal[i])
            }
            return false
        }
    }

    return true
}

public func printIfNotEqual<T: Numeric>(_ outMetal: T, _ outRef: T, _ pr: Bool = true, _ prNote: String = "") -> Bool
{
    if outMetal != outRef
    {
        if pr
        {
            print(prNote, "Error: ref = ", outRef, "; got out = ", outMetal)
        }
        return false
    }
    return true
}

public func printIfNotNear<T: BinaryFloatingPoint>(_ outMetal: [ T ], _ outRef: [ T ], _ N: Int, _ pr: Bool = true, _ prNote: String = "") -> Bool
{
    if outMetal.count != outRef.count
    {
        print("Error: Trying to compare arrays of different sizes, our size: ", outMetal.count, " ref size: ", outRef.count)
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
            if pr
            {
                print(prNote, "Error: ref[", i, " ] = ", outRefi, "; got out[", i, "] = ", outMeti)
            }
            return false
        }
    }

    return true
}

public func printIfNotNear<T: BinaryFloatingPoint>(_ outMetal: T, _ outRef: T, _ N : Int, _ pr : Bool = true, _ prNote : String = "") -> Bool
{
    let epsilon = 2 * (outRef.nextUp - outRef)
    let lowerBound = outRef - epsilon * sqrt(T(N))
    let upperBound = outRef + epsilon * sqrt(T(N))
    let rg : ClosedRange = lowerBound...upperBound
    
    if !(rg ~= outMetal)
    {
        if pr
        {
            print(prNote, "Error: ref = ", outRef, "; got out = ", outMetal)
        }
        return false
    }
    return true
}

public func initRandomVal<T: BinaryFloatingPoint>(_ val: inout T, _ range: ClosedRange<T> = (0...10), _ initType: InitType = .Integer)
where T.RawSignificand: FixedWidthInteger
{
    let neg = Bool.random()
    let tmp = (initType == .FloatingPoint) ? T(T.random(in: range))
                                           : T(Int.random(in: (Int(range.lowerBound)...Int(range.upperBound))))
    val = neg ? -tmp : tmp
}

public func initRandomVec<T: BinaryFloatingPoint>(_ arr: inout [T], _ N: Int, _ inc: Int, _ range: ClosedRange<T> = (0...10), _ initType: InitType = .Integer)
where T.RawSignificand: FixedWidthInteger
{
    // TODO: negative increments
    for _ in 0..<(N * inc)
    {
        // initializing between increments so we check that we aren't clobbering that data
        // Note: initTYpe == .DiagDomninant doesn't make sense in vector initialization, should handle this better
        let neg = Bool.random()
        let val = (initType == .FloatingPoint) ? T(T.random(in: range))
                                               : T(Int.random(in: (Int(range.lowerBound)...Int(range.upperBound))))
        arr.append(neg ? -val : val)
    }
}

public func initRandomVec<T: SignedInteger>(_ arr: inout [T], _ N: Int, _ inc: Int, _ range: ClosedRange<Int> = (0...10), _ initType: InitType = .Integer)
{
    for _ in 0..<(N * inc)
    {
        let neg = Bool.random()
        let val = T(Int.random(in: range))
        arr.append(neg ? -val : val)
    }
}

public func initRandomMatrix<T: BinaryFloatingPoint>(_ mat: inout [T], _ M: Int, _ N: Int, _ ld: Int, _ order: OrderType, _ range: ClosedRange<T> = (0...10), _ initType: InitType = .Integer, _ uplo: UploType = .FillLower)
where T.RawSignificand: FixedWidthInteger
{
    let size = order == .ColMajor ? ld * N : ld * M
    for _ in 0..<size
    {
        let neg = Bool.random()
        let val = (initType == .FloatingPoint) ? T(T.random(in: range))
                                               : T(Int.random(in: (Int(range.lowerBound)...Int(range.upperBound))))
        mat.append(neg ? -val : val)
    }

    if(initType == .DiagDominant)
    {
        // Row-diagonally dominant, provides numerical stability for things like triangular solves
        // For now DiagDominant implies a triangular matrix, so passing in uplo for this case
        // TODO: transpose?
        if uplo == .FillLower
        {
            for row in 0...M - 1
            {
                var sum : T = 0
                if(row > 0)
                {
                    for col in 0...row - 1
                    {
                        sum += abs(order == .ColMajor ? mat[col * ld + row] : mat[row * ld + col])
                    }
                }
                mat[row * ld + row] = sum == 0 ? 1 : sum * 2
            }
        }
        else
        {
            for row in 0...M - 1
            {
                var sum : T = 0
                if(row > 0)
                {
                    for col in row...N - 1
                    {
                        sum += abs(order == .ColMajor ? mat[col * ld + row] : mat[row * ld + col])
                    }
                }
                mat[row * ld + row] = sum == 0 ? 1 : sum * 2
            }
        }
    }
}

public func printVector<T: BinaryFloatingPoint>(_ arr: [T], _ N: Int, _ incx: Int)
{
    for i in 0..<N
    {
        print(arr[i], terminator: " ")
    }
    print("")
}

public func printMatrix<T: BinaryFloatingPoint>(_ arr: [T], _ M: Int, _ N: Int, _ lda: Int, _ order: OrderType)
{
    for i in 0..<M
    {
        for j in 0..<N
        {
            let idx = order == .ColMajor ? j * lda + i : i * lda + j
            print(arr[idx], terminator: " ")
        }
        print("")
    }

}
