//
//  metalGbmv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-03.
//

import Metal

public extension MetalBlas
{
    private func metalXgbmv<T: BinaryFloatingPoint>(_ order: OrderType, _ trans: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: T, _ matA: [T], _ lda: Int, _ vecx: [T], _ incx: Int, _ beta: T, _ vecy: inout [T], _ incy: Int)
    {
        let sizeA = order == .ColMajor ? N * lda : M * lda
        let sizex = trans == .NoTranspose ? N * incx : M * incx
        let sizey = trans == .NoTranspose ? M * incy : N * incy
        guard
            let bufferA = device.makeBuffer(bytes: matA, length: MemoryLayout<T>.stride * sizeA, options: [.storageModeManaged]),
            let bufferx = device.makeBuffer(bytes: vecx, length: MemoryLayout<T>.stride * sizex, options: [.storageModeManaged]),
            var buffery = device.makeBuffer(bytes: vecy, length: MemoryLayout<T>.stride * sizey, options: [.storageModeShared])
        else {
            fatalError("Failed to create buffers for matrices in gemv")
        }

        metalXgbmv(order, trans, M, N, KL, KU, alpha, bufferA, lda, bufferx, incx, beta, &buffery, incy)

        // Copy the data into the array
        copyBufToArray(buffery, &vecy)
    }

    private func metalXgbmv<T: BinaryFloatingPoint>(_ order: OrderType, _ trans: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: T, _ bufferA: MTLBuffer, _ lda: Int, _ bufferx: MTLBuffer, _ incx: Int, _ beta: T, _ buffery: inout MTLBuffer, _ incy: Int)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcString = T.self == Float.self ? "sgbmv" : "hgbmv"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function gbmv")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        var Mm = M, Nm = N, KLm = KL, KUm = KU, ldam = lda, alpham = alpha, betam = beta, incxm = incx, incym = incy
        var orderm = order, transm = trans
        let bufferM = device.makeBuffer(bytes: &Mm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferN = device.makeBuffer(bytes: &Nm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferKL = device.makeBuffer(bytes: &KLm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferKU = device.makeBuffer(bytes: &KUm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferlda = device.makeBuffer(bytes: &ldam, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncx = device.makeBuffer(bytes: &incxm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncy = device.makeBuffer(bytes: &incym, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferAlpha = device.makeBuffer(bytes: &alpham, length: MemoryLayout<T>.stride, options: .storageModePrivate)
        let bufferBeta = device.makeBuffer(bytes: &betam, length: MemoryLayout<T>.stride, options: .storageModePrivate)

        let bufferOrder = device.makeBuffer(bytes: &orderm, length: MemoryLayout<OrderType>.stride, options: .storageModePrivate)
        let bufferTrans = device.makeBuffer(bytes: &transm, length: MemoryLayout<TransposeType>.stride, options: .storageModePrivate)

        computeEncoder.setBuffer(bufferOrder, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferTrans, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferM, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferN, offset: 0, index: 3)
        computeEncoder.setBuffer(bufferKL, offset: 0, index: 4)
        computeEncoder.setBuffer(bufferKU, offset: 0, index: 5)
        computeEncoder.setBuffer(bufferAlpha, offset: 0, index: 6)
        computeEncoder.setBuffer(bufferA, offset: 0, index: 7)
        computeEncoder.setBuffer(bufferlda, offset: 0, index: 8)
        computeEncoder.setBuffer(bufferx, offset: 0, index: 9)
        computeEncoder.setBuffer(bufferIncx, offset: 0, index: 10)
        computeEncoder.setBuffer(bufferBeta, offset:0, index: 11)
        computeEncoder.setBuffer(buffery, offset: 0, index: 12)
        computeEncoder.setBuffer(bufferIncy, offset: 0, index: 13)

        let w = pipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)

        let K = trans == .NoTranspose ? M : N
        let threadgroups = MTLSize(
            width: K / w + 1,
            height: 1,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func metalHgbmv(_ order: OrderType, _ trans: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Float16, _ bufferA: MTLBuffer, _ lda: Int, _ bufferx: MTLBuffer, _ incx: Int, _ beta: Float16, _ buffery: inout MTLBuffer, _ incy: Int)
    {
        metalXgbmv(order, trans, M, N, KL, KU, alpha, bufferA, lda, bufferx, incx, beta, &buffery, incy)
    }

    func metalSgbmv(_ order: OrderType, _ trans: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Float, _ bufferA: MTLBuffer, _ lda: Int, _ bufferx: MTLBuffer, _ incx: Int, _ beta: Float, _ buffery: inout MTLBuffer, _ incy: Int)
    {
        metalXgbmv(order, trans, M, N, KL, KU, alpha, bufferA, lda, bufferx, incx, beta, &buffery, incy)
    }

    func metalDgbmv(_ order: OrderType, _ trans: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Double, _ bufferA: MTLBuffer, _ lda: Int, _ bufferx: MTLBuffer, _ incx: Int, _ beta: Double, _ buffery: inout MTLBuffer, _ incy: Int)
    {
        metalXgbmv(order, trans, M, N, KL, KU, alpha, bufferA, lda, bufferx, incx, beta, &buffery, incy)
    }

    func metalHgbmv(_ order: OrderType, _ trans: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Float16, _ matA: [Float16], _ lda: Int, _ vecx: [Float16], _ incx: Int, _ beta: Float16, _ vecy: inout [Float16], _ incy: Int)
    {
        metalXgbmv(order, trans, M, N, KL, KU, alpha, matA, lda, vecx, incx, beta, &vecy, incy)
    }

    func metalSgbmv(_ order: OrderType, _ trans: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Float, _ matA: [Float], _ lda: Int, _ vecx: [Float], _ incx: Int, _ beta: Float, _ vecy: inout [Float], _ incy: Int)
    {
        metalXgbmv(order, trans, M, N, KL, KU, alpha, matA, lda, vecx, incx, beta, &vecy, incy)
    }

    func metalDgbmv(_ order: OrderType, _ trans: TransposeType, _ M: Int, _ N: Int, _ KL: Int, _ KU: Int, _ alpha: Double, _ matA: [Double], _ lda: Int, _ vecx: [Double], _ incx: Int, _ beta: Double, _ vecy: inout [Double], _ incy: Int)
    {
        metalXgbmv(order, trans, M, N, KL, KU, alpha, matA, lda, vecx, incx, beta, &vecy, incy)
    }
}
