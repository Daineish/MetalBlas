//
//  referenceSpmv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-09.
//

import Accelerate

func cblasSpmv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ alpha: Float, _ A: [Float], _ x: [Float], _ incx: __LAPACK_int, _ beta: Float, _ y: inout [Float], _ incy: __LAPACK_int)
{
    cblas_sspmv(order, uplo, N, alpha, A, x, incx, beta, &y, incy)
}

func cblasSpmv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ alpha: Double, _ A: [Double], _ x: [Double], _ incx: __LAPACK_int, _ beta: Double, _ y: inout [Double], _ incy: __LAPACK_int)
{
    cblas_dspmv(order, uplo, N, alpha, A, x, incx, beta, &y, incy)
}

public func refHspmv(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float16, _ A: [ Float16 ], _ x: [ Float16 ], _ incx: Int, _ beta: Float16, _ y: inout [Float16], _ incy: Int)
{
    if N == 0 || (alpha == 0 && beta == 1)
    {
        return
    }

    for row in 0..<N
    {
        var sum : Float = 0
        for col in 0..<N
        {
            var r = row
            var c = col
            if (uplo == .FillUpper && row > col) || (uplo == .FillLower && row < col)
            {
                swap(&r, &c)
            }

            if uplo == .FillUpper
            {
                let ApIdx = order == .ColMajor ? c * (c + 1) / 2 + r : r * (N + 1) - (r * (r + 1)) / 2 + (c - r)
                sum += Float(A[ApIdx] * x[col * incx])
            }
            else
            {
                let ApIdx = order == .ColMajor ? c * (N + 1) - (c * (c + 1)) / 2 + (r - c) : r * (r + 1) / 2 + c
                sum += Float(A[ApIdx] * x[col * incx])
            }
        }

        y[row * incy] = alpha * Float16(sum) + beta * y[row * incy]
    }
}

public func refSspmv(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float, _ A: [ Float ], _ x: [ Float ], _ incx: Int, _ beta: Float, _ y: inout [Float], _ incy: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasSpmv(order_la, uplo_la , N_la, alpha, A, x, incx_la, beta, &y, incy_la)
}

public func refDspmv(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Double, _ A: [ Double ], _ x: [ Double ], _ incx: Int, _ beta: Double, _ y: inout [Double], _ incy: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasSpmv(order_la, uplo_la, N_la, alpha, A, x, incx_la, beta, &y, incy_la)
}
