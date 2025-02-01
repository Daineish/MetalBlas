//
//  referenceGemm.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Accelerate

// cblas gemm
func testSgemm(_ transA: CBLAS_TRANSPOSE, _ transB: CBLAS_TRANSPOSE, _ M: __LAPACK_int, _ N: __LAPACK_int, _ K: __LAPACK_int, _ alpha: Float, _ A: [Float], _ lda: __LAPACK_int, _ B: [Float], _ ldb: __LAPACK_int, _ beta: Float, _ C: inout [Float], _ ldc: __LAPACK_int)
{
    cblas_sgemm(CblasColMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, &C, ldc)
}

func testDgemm(_ transA: CBLAS_TRANSPOSE, _ transB: CBLAS_TRANSPOSE, _ M: Int, _ N: Int, _ K: Int, _ alpha: Double, _ A: [Double], _ lda: __LAPACK_int, _ B: [Double], _ ldb: __LAPACK_int, _ beta: Double, _ C: inout [Double], _ ldc: __LAPACK_int)
{
    cblas_dgemm(CblasColMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, &C, ldc)
}

// my gemm
public func myRefGemm<T: BinaryFloatingPoint>(_ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int,
             _ alpha: T, _ matA: [T], _ lda: Int, _ matB: [T], _ ldb: Int, _ beta: T, _ matC: inout [T], _ ldc: Int)
{
    // Only NN rn
    if(transA != .NoTranspose || transB != .NoTranspose)
    {
        return
    }

    for i in 0...M - 1
    {
        for j in 0...N - 1
        {
            let Cidx = j * ldc + i
            var tmp: T = 0
            for k in 0...K - 1
            {
                let Aidx = k * lda + i
                let Bidx = j * ldb + k
                tmp += matA[Aidx] * matB[Bidx]
            }
            matC[Cidx] = alpha * tmp + beta * matC[Cidx]
        }
    }
}

public func refHgemm(_ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int, _ alpha: Float16, _ A: [Float16], _ lda: Int, _ B: [Float16], _ ldb: Int, _ beta: Float16, _ C: inout [Float16], _ ldc: Int)
{
    myRefGemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, &C, ldc)
}

public func refSgemm(_ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int, _ alpha: Float, _ A: [Float], _ lda: Int, _ B: [Float], _ ldb: Int, _ beta: Float, _ C: inout [Float], _ ldc: Int)
{
    let M_la = __LAPACK_int(M)
    let N_la = __LAPACK_int(N)
    let K_la = __LAPACK_int(K)
    let lda_la = __LAPACK_int(lda)
    let ldb_la = __LAPACK_int(ldb)
    let ldc_la = __LAPACK_int(ldc)
    let transA_cblas = cblasTrans(transA)
    let transB_cblas = cblasTrans(transB)
    testSgemm(transA_cblas, transB_cblas, M_la, N_la, K_la, alpha, A, lda_la, B, ldb_la, beta, &C, ldc_la)
}

public func refDgemm(_ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int, _ alpha: Double, _ A: [Double], _ lda: Int, _ B: [Double], _ ldb: Int, _ beta: Double, _ C: inout [Double], _ ldc: Int)
{
    let M_la = __LAPACK_int(M)
    let N_la = __LAPACK_int(N)
    let K_la = __LAPACK_int(K)
    let lda_la = __LAPACK_int(lda)
    let ldb_la = __LAPACK_int(ldb)
    let ldc_la = __LAPACK_int(ldc)
    let transA_cblas = cblasTrans(transA)
    let transB_cblas = cblasTrans(transB)
    testDgemm(transA_cblas, transB_cblas, M_la, N_la, K_la, alpha, A, lda_la, B, ldb_la, beta, &C, ldc_la)
}

