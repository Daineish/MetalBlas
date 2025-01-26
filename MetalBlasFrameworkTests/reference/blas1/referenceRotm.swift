//
//  referenceRotm.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-26.
//

import Accelerate

func testRotm(_ N: __LAPACK_int, _ X: inout [Float], _ incx: __LAPACK_int, _ Y: inout [Float], _ incy: __LAPACK_int, _ P: [ Float ])
{
    cblas_srotm(N, &X, incx, &Y, incy, P)
}

func testRotm(_ N: __LAPACK_int, _ X: inout [Double], _ incx: __LAPACK_int, _ Y: inout [Double], _ incy: __LAPACK_int, _ P: [ Double ])
{
    cblas_drotm(N, &X, incx, &Y, incy, P)
}

public func refHrotm(_ N: Int, _ X: inout [ Float16 ], _ incx: Int, _ Y: inout [ Float16 ], _ incy: Int, _ P: [ Float16 ])
{
    // fine with fp comparison since these are exact representations
    let flag = P[0]
    if flag == -2.0
    {
        return
    }

    if flag != -1.0 && flag != 1.0 && flag != 0.0
    {
        // Error?
        return
    }

    let sh11 : Float = Float((flag == -1.0 || flag == 1.0) ? P[1] : 1.0)
    let sh21 : Float = Float((flag == -1.0 || flag == 0.0) ? P[2] : -1.0)
    let sh12 : Float = Float((flag == -1.0 || flag == 0.0) ? P[3] : 1.0)
    let sh22 : Float = Float((flag == -1.0 || flag == 1.0) ? P[4] : 1.0)

    for i in 0...N-1
    {
        let tmpx = Float(X[i * incx])
        let tmpy = Float(Y[i * incy])
        X[i * incx] = Float16(tmpx * sh11 + tmpy * sh12)
        Y[i * incy] = Float16(tmpx * sh21 + tmpy * sh22)
    }
}

public func refSrotm(_ N: Int, _ X: inout [ Float ], _ incx: Int, _ Y: inout [ Float ], _ incy: Int, _ P: [ Float ])
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testRotm(N_la, &X, incx_la, &Y, incy_la, P)
}

public func refDrotm(_ N: Int, _ X: inout [ Double ], _ incx: Int, _ Y: inout [ Double ], _ incy: Int, _ P: [ Double ])
{
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    testRotm(N_la, &X, incx_la, &Y, incy_la, P)
}
