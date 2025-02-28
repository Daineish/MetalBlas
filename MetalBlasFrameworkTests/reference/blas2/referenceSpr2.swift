//
//  referenceSpr2.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-22.
//

import Accelerate

func cblasSpr2(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ alpha: Float, _ x: [Float], _ incx: __LAPACK_int, _ y: [Float], _ incy: __LAPACK_int, _ A: inout [Float])
{
    cblas_sspr2(order, uplo, N, alpha, x, incx, y, incy, &A)
}

func cblasSpr2(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ alpha: Double, _ x: [Double], _ incx: __LAPACK_int, _ y: [Double], _ incy: __LAPACK_int, _ A: inout[Double])
{
    cblas_dspr2(order, uplo, N, alpha, x, incx, y, incy, &A)
}

public func refHspr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float16, _ x: [ Float16 ], _ incx: Int, _ y: [ Float16 ], _ incy: Int, _ A: inout [ Float16 ])
{
    if N == 0 || (alpha == 0)
    {
        return
    }

    if uplo == .FillUpper
    {
        for row in 0..<N
        {
            let valx = alpha * x[row * incx]
            let valy = alpha * y[row * incy]
            for col in row..<N
            {
                let ApIdx = order == .ColMajor ? col * (col + 1) / 2 + row : row * (N + 1) - (row * (row + 1)) / 2 + (col - row)
                A[ApIdx] += valx * y[col * incy] + valy * x[col * incx]
            }
        }
    }
    else
    {
        for row in 0..<N
        {
            let valx = alpha * x[row * incx]
            let valy = alpha * y[row * incy]
            for col in 0...row
            {
                let ApIdx = order == .ColMajor ? col * (N + 1) - (col * (col + 1)) / 2 + (row - col) : row * (row + 1) / 2 + col
                A[ApIdx] += valx * y[col * incy] + valy * x[col * incx]
            }
        }
    }
}

public func refSspr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float, _ x: [ Float ], _ incx: Int, _ y: [ Float ], _ incy: Int, _ A: inout [ Float ])
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasSpr2(order_la, uplo_la , N_la, alpha, x, incx_la, y, incy_la, &A)
}

public func refDspr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Double, _ x: [ Double ], _ incx: Int, _ y: [ Double ], _ incy: Int, _ A: inout [ Double ])
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasSpr2(order_la, uplo_la, N_la, alpha, x, incx_la, y, incy_la, &A)
}
