//
//  referenceTbmv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

import Accelerate

func cblasTbmv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ trans: CBLAS_TRANSPOSE, _ diag: CBLAS_DIAG, _ N: __LAPACK_int, _ K: __LAPACK_int, _ A: [Float], _ lda: __LAPACK_int, _ x: inout [Float], _ incx: __LAPACK_int)
{
    cblas_stbmv(order, uplo, trans, diag, N, K, A, lda, &x, incx)
}

func cblasTbmv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ trans: CBLAS_TRANSPOSE, _ diag: CBLAS_DIAG, _ N: __LAPACK_int, _ K: __LAPACK_int, _ A: [Double], _ lda: __LAPACK_int, _ x: inout [Double], _ incx: __LAPACK_int)
{
    cblas_dtbmv(order, uplo, trans, diag, N, K, A, lda, &x, incx)
}

public func refHtbmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ K: Int, _ A: [ Float16 ], _ lda: Int, _ x: inout [ Float16 ], _ incx: Int)
{
    if N == 0
    {
        return
    }

    let accessUpper = (uplo == .FillUpper && trans == .NoTranspose) || (uplo == .FillLower && trans != .NoTranspose)

    for r in 0..<N
    {
        let row = accessUpper ? r : N - 1 - r
        var sum : Float16 = 0

        if accessUpper
        {
            let minCol = row;
            let maxCol = min(N, row + K + 1)
            if(maxCol > minCol)
            {
                for col in minCol..<maxCol
                {
                    if trans == .NoTranspose
                    {
                        let AbIdx = order == .RowMajor ? (row * lda + (col - row)) : (col * lda + (row - col + K))
                        sum += (row == col && diag == .Unit) ? x[col * incx] : A[AbIdx] * x[col * incx]
                    }
                    else
                    {
                        let AbIdx = order == .RowMajor ? (col * lda + (row - col + K)) : (row * lda + (col - row))
                        sum += (row == col && diag == .Unit) ? x[col * incx] : A[AbIdx] * x[col * incx]
                    }
                }
            }
        }
        else
        {
            let minCol = 0 > row - K ? 0 : row - K
            let maxCol = row + 1
            if(maxCol > minCol)
            {
                for col in minCol..<maxCol
                {
                    if trans == .NoTranspose
                    {
                        let AbIdx = order == .RowMajor ? (row * lda + (col - row + K)) : (col * lda + (row - col))
                        sum += (row == col && diag == .Unit) ? x[col * incx] : A[AbIdx] * x[col * incx]
                    }
                    else
                    {
                        let AbIdx = order == .RowMajor ? (col * lda + (row - col)) : (row * lda + (col - row + K))
                        sum += (row == col && diag == .Unit) ? x[col * incx] : A[AbIdx] * x[col * incx]
                    }
                }
            }
        }

        x[row * incx] = sum
    }
}

public func refStbmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ K: Int, _ A: [ Float ], _ lda: Int, _ x: inout [ Float ], _ incx: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let trans_la = cblasTrans(trans)
    let diag_la = cblasDiag(diag)
    let N_la = __LAPACK_int(N)
    let K_la = __LAPACK_int(K)
    let incx_la = __LAPACK_int(incx)
    let lda_la = __LAPACK_int(lda)
    cblasTbmv(order_la, uplo_la, trans_la, diag_la, N_la, K_la, A, lda, &x, incx_la)
}

public func refDtbmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ K: Int, _ A: [ Double ], _ lda: Int, _ x: inout [ Double ], _ incx: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let trans_la = cblasTrans(trans)
    let diag_la = cblasDiag(diag)
    let N_la = __LAPACK_int(N)
    let K_la = __LAPACK_int(K)
    let incx_la = __LAPACK_int(incx)
    let lda_la = __LAPACK_int(lda)
    cblasTbmv(order_la, uplo_la, trans_la, diag_la, N_la, K_la, A, lda, &x, incx_la)
}
