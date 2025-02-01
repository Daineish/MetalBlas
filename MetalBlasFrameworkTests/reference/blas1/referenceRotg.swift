//
//  referenceRotg.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-26.
//

import Accelerate

public func refHrotg(_ a: inout Float16, _ b: inout Float16, _ c: inout Float16, _ s: inout Float16)
{
    let scale = abs(a) + abs(b)
    if scale == 0.0
    {
        c = 1.0
        s = 0.0
        a = 0.0
        b = 0.0
    }
    else
    {
        var roe = b
        if abs(a) > abs(b)
        {
            roe = a
        }

        let tmpA = a / scale
        let tmpB = b / scale
        var r = scale * sqrt(tmpA * tmpA + tmpB * tmpB)
        r = Float16(copysign(Float(r), Float(roe)))
        c = a / r
        s = b / r
        var z : Float16 = 1.0
        if abs(a) > abs(b)
        {
            z = s
        }

        if abs(b) >= abs(a) && c != 0.0
        {
            z = 1.0 / c
        }

        a = r
        b = z
    }
}

public func refSrotg(_ a: inout Float, _ b: inout Float, _ c: inout Float, _ s: inout Float)
{
    cblas_srotg(&a, &b, &c, &s)
}

public func refDrotg(_ a: inout Double, _ b: inout Double, _ c: inout Double, _ s: inout Double)
{
    cblas_drotg(&a, &b, &c, &s)
}
