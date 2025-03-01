//
//  referenceSyr2.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

import Accelerate

func cblasSyr2(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ alpha: Float, _ x: [Float], _ incx: __LAPACK_int, _ y: [Float], _ incy: __LAPACK_int, _ A: inout [Float], _ lda: Int)
{
    cblas_ssyr2(order, uplo, N, alpha, x, incx, y, incy, &A, lda)
}

func cblasSyr2(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ alpha: Double, _ x: [Double], _ incx: __LAPACK_int, _ y: [Double], _ incy: __LAPACK_int, _ A: inout[Double], _ lda: Int)
{
    cblas_dsyr2(order, uplo, N, alpha, x, incx, y, incy, &A, lda)
}

public func refHsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float16, _ x: [ Float16 ], _ incx: Int, _ y: [ Float16 ], _ incy: Int, _ A: inout [ Float16 ], _ lda: Int)
{
    if N == 0 || (alpha == 0)
    {
        return
    }

    for row in 0..<N
    {
        let valx = alpha * x[row * incx]
        let valy = alpha * y[row * incy]
        let colMin = uplo == .FillUpper ? row : 0
        let colMax = uplo == .FillUpper ? N : row + 1
        for col in colMin..<colMax
        {
            let AIdx = order == .ColMajor ? col * lda + row : row * lda + col
            A[AIdx] += valx * y[col * incy] + valy * x[col * incx]
        }
    }
}

public func refSsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float, _ x: [ Float ], _ incx: Int, _ y: [ Float ], _ incy: Int, _ A: inout [ Float ], _ lda: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    let lda_la = __LAPACK_int(lda)
    cblasSyr2(order_la, uplo_la , N_la, alpha, x, incx_la, y, incy_la, &A, lda_la)
}

public func refDsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Double, _ x: [ Double ], _ incx: Int, _ y: [ Double ], _ incy: Int, _ A: inout [ Double ], _ lda: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    let lda_la = __LAPACK_int(lda)
    cblasSyr2(order_la, uplo_la, N_la, alpha, x, incx_la, y, incy_la, &A, lda)
}
