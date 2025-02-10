//
//  referenceSbmv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-09.
//

import Accelerate

func cblasSbmv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ K: __LAPACK_int, _ alpha: Float, _ A: [Float], _ lda: __LAPACK_int, _ x: [Float], _ incx: __LAPACK_int, _ beta: Float, _ y: inout [Float], _ incy: __LAPACK_int)
{
    cblas_ssbmv(order, uplo, N, K, alpha, A, lda, x, incx, beta, &y, incy)
}

func cblasSbmv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ N: __LAPACK_int, _ K: __LAPACK_int, _ alpha: Double, _ A: [Double], _ lda: __LAPACK_int, _ x: [Double], _ incx: __LAPACK_int, _ beta: Double, _ y: inout [Double], _ incy: __LAPACK_int)
{
    cblas_dsbmv(order, uplo, N, K, alpha, A, lda, x, incx, beta, &y, incy)
}

public func refHsbmv(_ order: OrderType, _ uplo: UploType, _ N: Int, _ K: Int, _ alpha: Float16, _ A: [ Float16 ], _ lda: Int, _ x: [ Float16 ], _ incx: Int, _ beta: Float16, _ y: inout [Float16], _ incy: Int)
{
    if N == 0 || (alpha == 0 && beta == 1)
    {
        return
    }

    for row in 0..<N
    {
        var res : Float16 = 0
        let minCol = max(0, row - K)
        let maxCol = min(N, row + K + 1)
        if(maxCol > minCol)
        {
            for col in minCol..<maxCol
            {
                var c = col
                var r = row
                if (uplo == .FillUpper && row > col) || (uplo == .FillLower && col > row)
                {
                    swap(&c, &r)
                }

                let AbIdx = order == .RowMajor ? (r * lda + (c - r + (uplo == .FillLower ? K : 0))) : (c * lda + (r - c + (uplo == .FillUpper ? K : 0)))
                res += A[AbIdx] * x[col * incx]
            }
        }

        y[row * incy] = alpha * res + beta * y[row * incy]
    }
}

public func refSsbmv(_ order: OrderType, _ uplo: UploType, _ N: Int, _ K: Int, _ alpha: Float, _ A: [ Float ], _ lda: Int, _ x: [ Float ], _ incx: Int, _ beta: Float, _ y: inout [Float], _ incy: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let K_la = __LAPACK_int(K)
    let lda_la = __LAPACK_int(lda)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasSbmv(order_la, uplo_la , N_la, K_la, alpha, A, lda_la, x, incx_la, beta, &y, incy_la)
}

public func refDsbmv(_ order: OrderType, _ uplo: UploType, _ N: Int, _ K: Int, _ alpha: Double, _ A: [ Double ], _ lda: Int, _ x: [ Double ], _ incx: Int, _ beta: Double, _ y: inout [Double], _ incy: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let N_la = __LAPACK_int(N)
    let K_la = __LAPACK_int(K)
    let lda_la = __LAPACK_int(lda)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasSbmv(order_la, uplo_la, N_la, K_la, alpha, A, lda_la, x, incx_la, beta, &y, incy_la)
}
