//
//  referenceTbsv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-14.
//

import Accelerate

func cblasTbsv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ trans: CBLAS_TRANSPOSE, _ diag: CBLAS_DIAG, _ N: __LAPACK_int, _ K: __LAPACK_int, _ A: [Float], _ lda: __LAPACK_int, _ x: inout [Float], _ incx: __LAPACK_int)
{
    cblas_stbsv(order, uplo, trans, diag, N, K, A, lda, &x, incx)
}

func cblasTbsv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ trans: CBLAS_TRANSPOSE, _ diag: CBLAS_DIAG, _ N: __LAPACK_int, _ K: __LAPACK_int, _ A: [Double], _ lda: __LAPACK_int, _ x: inout [Double], _ incx: __LAPACK_int)
{
    cblas_dtbsv(order, uplo, trans, diag, N, K, A, lda, &x, incx)
}

public func refHtbsv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ K: Int, _ A: [ Float16 ], _ lda: Int, _ x: inout [ Float16 ], _ incx: Int)
{
    if N == 0
    {
        return
    }

    let accessUpper = (uplo == .FillUpper && trans == .NoTranspose) || (uplo == .FillLower && trans != .NoTranspose)

    for r in 0..<N
    {
        let row = accessUpper ? r : N - 1 - r
        var sum : Float16 = x[row * incx]

        if accessUpper
        {
            let minCol = row
            let maxCol = min(N, row + K + 1)
            if(maxCol > minCol)
            {
                for col in minCol..<maxCol
                {
                    if trans == .NoTranspose
                    {
                        let AbIdx = order == .RowMajor ? (row * lda + (col - row)) : (col * lda + (row - col + K))
                        sum -= (row == col && diag == .Unit) ? x[col * incx] : A[AbIdx] * x[col * incx]
                    }
                    else
                    {
                        let AbIdx = order == .RowMajor ? (col * lda + (row - col + K)) : (row * lda + (col - row))
                        sum -= (row == col && diag == .Unit) ? x[col * incx] : A[AbIdx] * x[col * incx]
                    }
                }
            }
        }
        else
        {
            let minCol = 0 > row - K ? 0 : row - K
            let maxCol = row
            if(maxCol > minCol)
            {
                for col in minCol..<maxCol
                {
                    if trans == .NoTranspose
                    {
                        let AbIdx = order == .RowMajor ? (row * lda + (col - row + K)) : (col * lda + (row - col))
                        sum -= A[AbIdx] * x[col * incx]
                    }
                    else
                    {
                        let AbIdx = order == .RowMajor ? (col * lda + (row - col)) : (row * lda + (col - row + K))
                        sum -= A[AbIdx] * x[col * incx]
                    }
                }
            }
        }

        if diag != .Unit
        {
            var AbIdx = row * lda
            if((uplo == .FillUpper && order == .ColMajor) || (uplo == .FillLower && order == .RowMajor))
            {
                AbIdx += K
            }
            sum /= A[AbIdx]
        }
        x[row * incx] = sum
    }
}

public func refStbsv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ K: Int, _ A: [ Float ], _ lda: Int, _ x: inout [ Float ], _ incx: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let trans_la = cblasTrans(trans)
    let diag_la = cblasDiag(diag)
    cblasTbsv(order_la, uplo_la, trans_la, diag_la, N, K, A, lda, &x, incx)
}

public func refDtbsv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ K: Int, _ A: [ Double ], _ lda: Int, _ x: inout [ Double ], _ incx: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let trans_la = cblasTrans(trans)
    let diag_la = cblasDiag(diag)
    cblasTbsv(order_la, uplo_la, trans_la, diag_la, N, K, A, lda, &x, incx)
}
