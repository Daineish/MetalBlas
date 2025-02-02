//
//  referenceGer.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

import Accelerate

func cblasGer(_ order: CBLAS_ORDER, _ M: __LAPACK_int, _ N: __LAPACK_int, _ alpha: Float, _ x: [Float], _ incx: __LAPACK_int, _ y: [Float], _ incy: __LAPACK_int, _ A: inout [Float], _ lda: __LAPACK_int)
{
    cblas_sger(order, M, N, alpha, x, incx, y, incy, &A, lda)
}

func cblasGer(_ order: CBLAS_ORDER, _ M: __LAPACK_int, _ N: __LAPACK_int, _ alpha: Double, _ x: [Double], _ incx: __LAPACK_int, _ y: [Double], _ incy: __LAPACK_int, _ A: inout [Double], _ lda: __LAPACK_int)
{
    cblas_dger(order, M, N, alpha, x, incx, y, incy, &A, lda)
}

public func refHger(_ order: OrderType, _ M: Int, _ N: Int, _ alpha: Float16, _ x: [ Float16 ], _ incx: Int, _ y: [Float16], _ incy: Int, _ A: inout [Float16], _ lda: Int)
{
    if M == 0 || N == 0 || alpha == 0
    {
        return
    }

    let lda1 = order == .ColMajor ? lda : 1
    let lda2 = order == .ColMajor ? 1 : lda

    for i in 0..<N
    {
        let yVal = alpha * y[i * incy]
        for j in 0..<M
        {
            A[i * lda1 + j * lda2] += yVal * x[j * incx]
        }
    }
}

public func refSger(_ order: OrderType, _ M: Int, _ N: Int, _ alpha: Float, _ x: [ Float ], _ incx: Int, _ y: [Float], _ incy: Int, _ A: inout [Float], _ lda: Int)
{
    let order_la = cblasOrder(order)
    let M_la = __LAPACK_int(M)
    let N_la = __LAPACK_int(N)
    let lda_la = __LAPACK_int(lda)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasGer(order_la, M_la, N_la, alpha, x, incx_la, y, incy_la, &A, lda_la)
}

public func refDger(_ order: OrderType, _ M: Int, _ N: Int, _ alpha: Double, _ x: [ Double ], _ incx: Int, _ y: [Double], _ incy: Int, _ A: inout [Double], _ lda: Int)
{
    let order_la = cblasOrder(order)
    let M_la = __LAPACK_int(M)
    let N_la = __LAPACK_int(N)
    let lda_la = __LAPACK_int(lda)
    let incx_la = __LAPACK_int(incx)
    let incy_la = __LAPACK_int(incy)
    cblasGer(order_la, M_la, N_la, alpha, x, incx_la, y, incy_la, &A, lda_la)
}
