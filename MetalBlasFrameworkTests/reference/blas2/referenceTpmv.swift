//
//  referenceTpmv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-06-21.
//

import Accelerate

func cblasTpmv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ trans: CBLAS_TRANSPOSE, _ diag: CBLAS_DIAG, _ N: __LAPACK_int, _ Ap: [Float], _ x: inout [Float], _ incx: __LAPACK_int)
{
    cblas_stpmv(order, uplo, trans, diag, N, Ap, &x, incx)
}

func cblasTpmv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ trans: CBLAS_TRANSPOSE, _ diag: CBLAS_DIAG, _ N: __LAPACK_int, _ Ap: [Double], _ x: inout [Double], _ incx: __LAPACK_int)
{
    cblas_dtpmv(order, uplo, trans, diag, N, Ap, &x, incx)
}

public func refHtpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ Ap: [ Float16 ], _ x: inout [ Float16 ], _ incx: Int)
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
            for col in row..<N
            {
                if trans == .NoTranspose
                {
                    let ApIdx = order == .RowMajor ? (row * N + col - row) - ((row - 1) * row) / 2 : (col * (col + 1) / 2 + row)
                    sum += (row == col && diag == .Unit) ? x[col * incx] : Ap[ApIdx] * x[col * incx]
                }
                else
                {
                    let ApIdx = order == .RowMajor ? (col * (col + 1) / 2 + row) : (row * N + col - row) - ((row - 1) * row) / 2
                    sum += (row == col && diag == .Unit) ? x[col * incx] : Ap[ApIdx] * x[col * incx]
                }
            }
        }
        else
        {
            for col in 0...row
            {
                if trans == .NoTranspose
                {
                    let ApIdx = order == .RowMajor ? (row * (row + 1) / 2 + col) : (col * N + row - col) - ((col - 1) * col) / 2
                    sum += (row == col && diag == .Unit) ? x[col * incx] : Ap[ApIdx] * x[col * incx]
                }
                else
                {
                    let ApIdx = order == .RowMajor ? (col * N + row - col) - ((col - 1) * col) / 2 : (row * (row + 1) / 2 + col)
                    sum += (row == col && diag == .Unit) ? x[col * incx] : Ap[ApIdx] * x[col * incx]
                }
            }
        }

        x[row * incx] = sum
    }
}

public func refStpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ A: [ Float ], _ x: inout [ Float ], _ incx: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let trans_la = cblasTrans(trans)
    let diag_la = cblasDiag(diag)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    cblasTpmv(order_la, uplo_la, trans_la, diag_la, N_la, A, &x, incx_la)
}

public func refDtpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ A: [ Double ], _ x: inout [ Double ], _ incx: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let trans_la = cblasTrans(trans)
    let diag_la = cblasDiag(diag)
    let N_la = __LAPACK_int(N)
    let incx_la = __LAPACK_int(incx)
    cblasTpmv(order_la, uplo_la, trans_la, diag_la, N_la, A, &x, incx_la)
}
