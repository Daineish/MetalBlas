//
//  referenceGemv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

import Accelerate

func cblasGemv(_ order: CBLAS_ORDER, _ trans: CBLAS_TRANSPOSE, _ M: __LAPACK_int, _ N: __LAPACK_int, _ alpha: Float, _ A: [Float], _ lda: __LAPACK_int, _ x: [Float], _ incx: __LAPACK_int, _ beta: Float, _ y: inout [Float], _ incy: __LAPACK_int)
{
    cblas_sgemv(order, trans, M, N, alpha, A, lda, x, incx, beta, &y, incy)
}

func cblasGemv(_ order: CBLAS_ORDER, _ trans: CBLAS_TRANSPOSE, _ M: __LAPACK_int, _ N: __LAPACK_int, _ alpha: Double, _ A: [Double], _ lda: __LAPACK_int, _ x: [Double], _ incx: __LAPACK_int, _ beta: Double, _ y: inout [Double], _ incy: __LAPACK_int)
{
    cblas_dgemv(order, trans, M, N, alpha, A, lda, x, incx, beta, &y, incy)
}

public func refHgemv(_ order: OrderType, _ transA: TransposeType, _ M: Int, _ N: Int, _ alpha: Float16, _ A: [ Float16 ], _ lda: Int, _ x: [ Float16 ], _ incx: Int, _ beta: Float16, _ y: inout [Float16], _ incy: Int)
{
    if M == 0 || N == 0 || (alpha == 0 && beta == 1)
    {
        return
    }

    let lda1 = order == .ColMajor ? lda : 1
    let lda2 = order == .ColMajor ? 1 : lda

    let dim1 = transA == .NoTranspose ? M : N
    let dim2 = transA == .NoTranspose ? N : M

    for i in 0..<dim1
    {
        var sum : Float = 0
        for j in 0..<dim2
        {
            let Aidx = transA == .NoTranspose ? j * lda1 + i * lda2 : i * lda1 + j * lda2
            sum += Float(A[Aidx] * x[j * incx])
        }
        y[i * incy] = alpha * Float16(sum) + beta * y[i * incy]
    }
}

public func refSgemv(_ order: OrderType, _ transA: TransposeType, _ M: Int, _ N: Int, _ alpha: Float, _ A: [ Float ], _ lda: Int, _ x: [ Float ], _ incx: Int, _ beta: Float, _ y: inout [Float], _ incy: Int)
{
    let order_la = cblasOrder(order)
    let trans_la = cblasTrans(transA)
    let M_la = __LAPACK_int(M)
    let N_la = __LAPACK_int(N)
    let lda_la = __LAPACK_int(lda)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasGemv(order_la, trans_la, M_la, N_la, alpha, A, lda_la, x, incx_la, beta, &y, incy_la)
}

public func refDgemv(_ order: OrderType, _ transA: TransposeType, _ M: Int, _ N: Int, _ alpha: Double, _ A: [ Double ], _ lda: Int, _ x: [ Double ], _ incx: Int, _ beta: Double, _ y: inout [Double], _ incy: Int)
{
    let order_la = cblasOrder(order)
    let trans_la = cblasTrans(transA)
    let M_la = __LAPACK_int(M)
    let N_la = __LAPACK_int(N)
    let lda_la = __LAPACK_int(lda)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasGemv(order_la, trans_la, M_la, N_la, alpha, A, lda_la, x, incx_la, beta, &y, incy_la)
}
