//
//  referenceGbmv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-02.
//

import Accelerate

func cblasGbmv(_ order: CBLAS_ORDER, _ trans: CBLAS_TRANSPOSE, _ M: __LAPACK_int, _ N: __LAPACK_int, _ KL: __LAPACK_int, _ KU: __LAPACK_int, _ alpha: Float, _ A: [Float], _ lda: __LAPACK_int, _ x: [Float], _ incx: __LAPACK_int, _ beta: Float, _ y: inout [Float], _ incy: __LAPACK_int)
{
    cblas_sgbmv(order, trans, M, N, KL, KU, alpha, A, lda, x, incx, beta, &y, incy)
}

func cblasGbmv(_ order: CBLAS_ORDER, _ trans: CBLAS_TRANSPOSE, _ M: __LAPACK_int, _ N: __LAPACK_int, _ KL: __LAPACK_int, _ KU: __LAPACK_int, _ alpha: Double, _ A: [Double], _ lda: __LAPACK_int, _ x: [Double], _ incx: __LAPACK_int, _ beta: Double, _ y: inout [Double], _ incy: __LAPACK_int)
{
    cblas_dgbmv(order, trans, M, N, KL, KU, alpha, A, lda, x, incx, beta, &y, incy)
}

public func refHgbmv(_ order: OrderType, _ transA: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Float16, _ A: [ Float16 ], _ lda: Int, _ x: [ Float16 ], _ incx: Int, _ beta: Float16, _ y: inout [Float16], _ incy: Int)
{
    if M == 0 || N == 0 || (alpha == 0 && beta == 1)
    {
        return
    }

    if(transA == .NoTranspose)
    {
        for row in 0..<M
        {
            var res : Float16 = 0
            let minCol = max(0, row - KL)
            let maxCol = min(N, row + KU + 1)
            if maxCol > minCol
            {
                for col in minCol..<maxCol
                {
                    let AbIdx = order == .RowMajor ? (row * lda + (col - row + KL)) : (col * lda + (row - col + KU))
                    res += A[AbIdx] * x[col * incx]
                }
            }
            y[row * incy] = alpha * res + beta * y[row * incy]
        }
    }
    else
    {
        for col in 0..<N
        {
            var res : Float16 = 0
            let minRow = max(0, col - KU)
            let maxRow = min(M, col + KL + 1)
            if maxRow > minRow
            {
                for row in minRow..<maxRow
                {
                    let AbIdx = order == .RowMajor ? (row * lda + (col - row + KL)) : (col * lda + (row - col + KU))
                    res += A[AbIdx] * x[row * incx]
                }
            }
            y[col * incy] = alpha * res + beta * y[col * incy]
        }
    }
}

public func refSgbmv(_ order: OrderType, _ transA: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Float, _ A: [ Float ], _ lda: Int, _ x: [ Float ], _ incx: Int, _ beta: Float, _ y: inout [Float], _ incy: Int)
{
    let order_la = cblasOrder(order)
    let trans_la = cblasTrans(transA)
    let M_la = __LAPACK_int(M)
    let N_la = __LAPACK_int(N)
    let KL_la = __LAPACK_int(KL)
    let KU_la = __LAPACK_int(KU)
    let lda_la = __LAPACK_int(lda)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasGbmv(order_la, trans_la, M_la, N_la, KL_la, KU_la, alpha, A, lda_la, x, incx_la, beta, &y, incy_la)
}

public func refDgbmv(_ order: OrderType, _ transA: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Double, _ A: [ Double ], _ lda: Int, _ x: [ Double ], _ incx: Int, _ beta: Double, _ y: inout [Double], _ incy: Int)
{
    let order_la = cblasOrder(order)
    let trans_la = cblasTrans(transA)
    let M_la = __LAPACK_int(M)
    let N_la = __LAPACK_int(N)
    let KL_la = __LAPACK_int(KL)
    let KU_la = __LAPACK_int(KU)
    let lda_la = __LAPACK_int(lda)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasGbmv(order_la, trans_la, M_la, N_la, KL_la, KU_la, alpha, A, lda_la, x, incx_la, beta, &y, incy_la)
}
