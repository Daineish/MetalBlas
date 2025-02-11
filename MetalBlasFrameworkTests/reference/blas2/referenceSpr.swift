//
//  referenceSpr.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-10.
//

import Accelerate

func cblasSpr(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ alpha: Float, _ x: [Float], _ incx: __LAPACK_int, _ A: inout [Float])
{
    cblas_sspr(order, uplo, N, alpha, x, incx, &A)
}

func cblasSpr(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ alpha: Double, _ x: [Double], _ incx: __LAPACK_int, _ A: inout[Double])
{
    cblas_dspr(order, uplo, N, alpha, x, incx, &A)
}

public func refHspr(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float16, _ x: [ Float16 ], _ incx: Int, _ A: inout [ Float16 ])
{
    if N == 0 || (alpha == 0)
    {
        return
    }

    if uplo == .FillUpper
    {
        for row in 0..<N
        {
            let val = alpha * x[row * incx]
            for col in row..<N
            {
                let ApIdx = order == .ColMajor ? col * (col + 1) / 2 + row : row * (N + 1) - (row * (row + 1)) / 2 + (col - row)
                A[ApIdx] += val * x[col * incx]
            }
        }
    }
    else
    {
        for row in 0..<N
        {
            let val = alpha * x[row * incx]
            for col in 0...row
            {
                let ApIdx = order == .ColMajor ? col * (N + 1) - (col * (col + 1)) / 2 + (row - col) : row * (row + 1) / 2 + col
                A[ApIdx] += val * x[col * incx]
            }
        }
    }
}

public func refSspr(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float, _ x: [ Float ], _ incx: Int, _ A: inout [ Float ])
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    cblasSpr(order_la, uplo_la , N_la, alpha, x, incx_la, &A)
}

public func refDspr(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Double, _ x: [ Double ], _ incx: Int, _ A: inout [ Double ])
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    cblasSpr(order_la, uplo_la, N_la, alpha, x, incx_la, &A)
}
