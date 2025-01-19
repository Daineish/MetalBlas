//
//  metalGemm.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Metal

public extension MetalBlas
{
    func metalXgemm<T: BinaryFloatingPoint>(_ storageMethod: StorageMethod, _ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int,
                           _ alpha: T, _ matA: [T], _ lda: Int, _ matB: [T], _ ldb: Int, _ beta: T, _ matC: inout [T], _ ldc: Int)
    {
        // TODO: potentially have smarter memory handling in this class
        guard
            let bufferA = device.makeBuffer(bytes: matA, length: MemoryLayout<T>.stride * K * lda, options: .storageModeManaged),
            let bufferB = device.makeBuffer(bytes: matB, length: MemoryLayout<T>.stride * N * ldb, options: .storageModeManaged),
            var bufferC = device.makeBuffer(bytes: matC, length: MemoryLayout<T>.stride * N * ldc, options: .storageModeShared)
        else {
            fatalError("Failed to create buffers for matrices in sgemm")
        }

        metalXgemm(storageMethod, transA, transB, M, N, K, alpha, bufferA, lda, bufferB, ldb, beta, &bufferC, ldc)

        copyBufToArray(bufferC, &matC)
    }

    func metalXgemm<T: BinaryFloatingPoint>(_ storageMethod: StorageMethod, _ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int,
                           _ alpha: T, _ matA: MTLBuffer, _ lda: Int, _ matB: MTLBuffer, _ ldb: Int, _ beta: T, _ matC: inout MTLBuffer, _ ldc: Int)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcString = T.self == Float.self ? "sgemm" : "hgemm"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function gemm")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        // Can change to bitmath if I get fancy
//        var transposeType: Int8 = (transA == .NoTranspose && transB == .NoTranspose) ? 1
//                          : (transA != .NoTranspose && transB == .NoTranspose) ? 2
//                          : (transA == .NoTranspose && transB != .NoTranspose) ? 3 : 4

        var Mm = M, Nm = N, Km = K, ldam = lda, ldbm = ldb, ldcm = ldc, alpham = alpha, betam = beta
//        let bufferTranspose = device.makeBuffer(bytes: &transposeType, length: MemoryLayout<Int8>.stride, options: .storageModePrivate)
        let bufferM = device.makeBuffer(bytes: &Mm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferN = device.makeBuffer(bytes: &Nm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferK = device.makeBuffer(bytes: &Km, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferLda = device.makeBuffer(bytes: &ldam, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferLdb = device.makeBuffer(bytes: &ldbm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferLdc = device.makeBuffer(bytes: &ldcm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferAlpha = device.makeBuffer(bytes: &alpham, length: MemoryLayout<T>.stride, options: .storageModePrivate)
        let bufferBeta = device.makeBuffer(bytes: &betam, length: MemoryLayout<T>.stride, options: .storageModePrivate)

//        computeEncoder.setBuffer(bufferTranspose, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferM, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferN, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferK, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferAlpha, offset: 0, index: 3)
        computeEncoder.setBuffer(matA, offset: 0, index: 4)
        computeEncoder.setBuffer(bufferLda, offset: 0, index: 5)
        computeEncoder.setBuffer(matB, offset: 0, index: 6)
        computeEncoder.setBuffer(bufferLdb, offset: 0, index: 7)
        computeEncoder.setBuffer(bufferBeta, offset: 0, index: 8)
        computeEncoder.setBuffer(matC, offset: 0, index: 9)
        computeEncoder.setBuffer(bufferLdc, offset: 0, index: 10)

        if(transA == .NoTranspose && transB == .NoTranspose)
        {
            let w = pipelineState.threadExecutionWidth
            let h = pipelineState.maxTotalThreadsPerThreadgroup / w
            let threadsPerThreadgroup = MTLSizeMake(w, h, 1)

            let threadgroups = MTLSize(
                width: (N / w) + 1,
                height: (M / h) + 1,
                depth: 1
            )

            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        else
        {
            fatalError("Other transpose types not yet supported")
        }
    }

    func metalHgemm(_ storageMethod: StorageMethod, _ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int,
                           _ alpha: Float16, _ matA: [Float16], _ lda: Int, _ matB: [Float16], _ ldb: Int, _ beta: Float16, _ matC: inout [Float16], _ ldc: Int)
    {
        metalXgemm(storageMethod, transA, transB, M, N, K, alpha, matA, lda, matB, ldb, beta, &matC, ldc)
    }

    func metalSgemm(_ storageMethod: StorageMethod, _ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int,
                           _ alpha: Float, _ matA: [Float], _ lda: Int, _ matB: [Float], _ ldb: Int, _ beta: Float, _ matC: inout [Float], _ ldc: Int)
    {
        metalXgemm(storageMethod, transA, transB, M, N, K, alpha, matA, lda, matB, ldb, beta, &matC, ldc)
    }

    func metalDgemm(_ storageMethod: StorageMethod, _ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int,
                           _ alpha: Double, _ matA: [Double], _ lda: Int, _ matB: [Double], _ ldb: Int, _ beta: Double, _ matC: inout [Double], _ ldc: Int)
    {
        metalXgemm(storageMethod, transA, transB, M, N, K, alpha, matA, lda, matB, ldb, beta, &matC, ldc)
    }

    func metalHgemm(_ storageMethod: StorageMethod, _ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int, _ alpha: Float16, _ matA: MTLBuffer, _ lda: Int, _ matB: MTLBuffer, _ ldb: Int, _ beta: Float16, _ matC: inout MTLBuffer, _ ldc: Int)
    {
        metalXgemm(storageMethod, transA, transB, M, N, K, alpha, matA, lda, matB, ldb, beta, &matC, ldc)
    }

    func metalSgemm(_ storageMethod: StorageMethod, _ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int, _ alpha: Float, _ matA: MTLBuffer, _ lda: Int, _ matB: MTLBuffer, _ ldb: Int, _ beta: Float, _ matC: inout MTLBuffer, _ ldc: Int)
    {
        metalXgemm(storageMethod, transA, transB, M, N, K, alpha, matA, lda, matB, ldb, beta, &matC, ldc)
    }

    func metalDgemm(_ storageMethod: StorageMethod, _ transA: TransposeType, _ transB: TransposeType, _ M: Int, _ N: Int, _ K: Int, _ alpha: Double, _ matA: MTLBuffer, _ lda: Int, _ matB: MTLBuffer, _ ldb: Int, _ beta: Double, _ matC: inout MTLBuffer, _ ldc: Int)
    {
        metalXgemm(storageMethod, transA, transB, M, N, K, alpha, matA, lda, matB, ldb, beta, &matC, ldc)
    }
}
