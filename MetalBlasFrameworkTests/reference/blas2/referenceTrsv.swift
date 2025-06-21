//
//  referenceTrsv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-06-21.
//

import Accelerate

func cblasTrsv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ trans: CBLAS_TRANSPOSE, _ diag: CBLAS_DIAG, _ N: __LAPACK_int, _ A: [Float], _ lda: __LAPACK_int, _ x: inout [Float], _ incx: __LAPACK_int)
{
    cblas_strsv(order, uplo, trans, diag, N, A, lda, &x, incx)
}

func cblasTrsv(_ order: CBLAS_ORDER, _ uplo: CBLAS_UPLO, _ trans: CBLAS_TRANSPOSE, _ diag: CBLAS_DIAG, _ N: __LAPACK_int, _ A: [Double], _ lda: __LAPACK_int, _ x: inout [Double], _ incx: __LAPACK_int)
{
    cblas_dtrsv(order, uplo, trans, diag, N, A, lda, &x, incx)
}

public func refHtrsv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ A: [ Float16 ], _ lda: Int, _ x: inout [ Float16 ], _ incx: Int)
{
    // TODO: this
    if N == 0
    {
        return
    }
}

public func refStrsv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ A: [ Float ], _ lda: Int, _ x: inout [ Float ], _ incx: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let trans_la = cblasTrans(trans)
    let diag_la = cblasDiag(diag)
    cblasTrsv(order_la, uplo_la, trans_la, diag_la, N, A, lda, &x, incx)
}

public func refDtrsv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType ,_ N: Int, _ A: [ Double ], _ lda: Int, _ x: inout [ Double ], _ incx: Int)
{
    let order_la = cblasOrder(order)
    let uplo_la = cblasUplo(uplo)
    let trans_la = cblasTrans(trans)
    let diag_la = cblasDiag(diag)
    cblasTrsv(order_la, uplo_la, trans_la, diag_la, N, A, lda, &x, incx)
}
