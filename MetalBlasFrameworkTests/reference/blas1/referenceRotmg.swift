//
//  referenceRotmg.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

import Accelerate

public func refHrotmg(_ d1: inout Float16, _ d2: inout Float16, _ b1: inout Float16, _ b2: Float16, _ P: inout [Float16])
{
    var d1F = Float(d1)
    var d2F = Float(d2)
    var b1F = Float(b1)
    let b2F = Float(b2)
    var pF : [Float] = []
    for i in 0..<P.count
    {
        pF.append(Float(P[i]))
    }

    // Not sure how this will handle overflow
    refSrotmg(&d1F, &d2F, &b1F, b2F, &pF)
    d1 = Float16(d1F)
    d2 = Float16(d2F)
    b1 = Float16(b1F)
    for i in 0..<P.count
    {
        P[i] = Float16(pF[i])
    }
}

public func refSrotmg(_ d1: inout Float, _ d2: inout Float, _ b1: inout Float, _ b2: Float, _ P: inout [Float])
{
    cblas_srotmg(&d1, &d2, &b1, b2, &P)
}

public func refDrotmg(_ d1: inout Double, _ d2: inout Double, _ b1: inout Double, _ b2: Double, _ P: inout [Double])
{
    cblas_drotmg(&d1, &d2, &b1, b2, &P)
}
