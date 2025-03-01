//
//  metalSyr2.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

import Metal

public extension MetalBlas
{
    private func metalXsyr2<T: BinaryFloatingPoint>(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: T, _ vecx: [T], _ incx: Int, _ vecy: [T], _ incy: Int, _ matA: inout [T], _ lda: Int)
    {
        let sizeA = N * lda
        let sizex = N * incx
        let sizey = N * incy
        guard
            var bufferA = device.makeBuffer(bytes: matA, length: MemoryLayout<T>.stride * sizeA, options: [.storageModeManaged]),
            let bufferx = device.makeBuffer(bytes: vecx, length: MemoryLayout<T>.stride * sizex, options: [.storageModeManaged]),
            let buffery = device.makeBuffer(bytes: vecy, length: MemoryLayout<T>.stride * sizey, options: [.storageModeManaged])
        else {
            fatalError("Failed to create buffers for matrices in syr2")
        }

        metalXsyr2(order, uplo, N, alpha, bufferx, incx, buffery, incy, &bufferA, lda)

        // Copy the data into the array
        copyBufToArray(bufferA, &matA)
    }

    private func metalXsyr2<T: BinaryFloatingPoint>(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: T, _ bufferx: MTLBuffer, _ incx: Int, _ buffery: MTLBuffer, _ incy: Int, _ bufferA: inout MTLBuffer, _ lda: Int)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcString = T.self == Float.self ? "ssyr2" : "hsyr2"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function syr2")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        var Nm = N, alpham = alpha, incxm = incx, incym = incy, ldam = lda
        var orderm = order, uplom = uplo
        let bufferN = device.makeBuffer(bytes: &Nm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncx = device.makeBuffer(bytes: &incxm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncy = device.makeBuffer(bytes: &incym, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferLda = device.makeBuffer(bytes: &ldam, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferAlpha = device.makeBuffer(bytes: &alpham, length: MemoryLayout<T>.stride, options: .storageModePrivate)

        let bufferOrder = device.makeBuffer(bytes: &orderm, length: MemoryLayout<OrderType>.stride, options: .storageModePrivate)
        let bufferUplo = device.makeBuffer(bytes: &uplom, length: MemoryLayout<UploType>.stride, options: .storageModePrivate)

        computeEncoder.setBuffer(bufferOrder, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferUplo, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferN, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferAlpha, offset: 0, index: 3)
        computeEncoder.setBuffer(bufferx, offset: 0, index: 4)
        computeEncoder.setBuffer(bufferIncx, offset: 0, index: 5)
        computeEncoder.setBuffer(buffery, offset: 0, index: 6)
        computeEncoder.setBuffer(bufferIncy, offset: 0, index: 7)
        computeEncoder.setBuffer(bufferA, offset: 0, index: 8)
        computeEncoder.setBuffer(bufferLda, offset: 0, index: 9)

        let w = pipelineState.maxTotalThreadsPerThreadgroup
        let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)

        let threadgroups = MTLSize(
            width: N / w + 1,
            height: 1,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func metalHsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float16, _ bufferx: MTLBuffer, _ incx: Int, _ buffery: MTLBuffer, _ incy: Int, _ bufferA: inout MTLBuffer, _ lda: Int)
    {
        metalXsyr2(order, uplo, N, alpha, bufferx, incx, buffery, incy, &bufferA, lda)
    }

    func metalSsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float, _ bufferx: MTLBuffer, _ incx: Int, _ buffery: MTLBuffer, _ incy: Int, _ bufferA: inout MTLBuffer, _ lda: Int)
    {
        metalXsyr2(order, uplo, N, alpha, bufferx, incx, buffery, incy, &bufferA, lda)
    }

    func metalDsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Double, _ bufferx: MTLBuffer, _ incx: Int, _ buffery: MTLBuffer, _ incy: Int, _ bufferA: inout MTLBuffer, _ lda: Int)
    {
        metalXsyr2(order, uplo, N, alpha, bufferx, incx, buffery, incy, &bufferA, lda)
    }

    func metalHsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float16, _ vecx: [Float16], _ incx: Int, _ vecy: [Float16], _ incy: Int, _ matA: inout [Float16], _ lda: Int)
    {
        metalXsyr2(order, uplo, N, alpha, vecx, incx, vecy, incy, &matA, lda)
    }

    func metalSsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Float, _ vecx: [Float], _ incx: Int, _ vecy: [Float], _ incy: Int, _ matA: inout[Float], _ lda: Int)
    {
        metalXsyr2(order, uplo, N, alpha, vecx, incx, vecy, incy, &matA, lda)
    }

    func metalDsyr2(_ order: OrderType, _ uplo: UploType, _ N: Int, _ alpha: Double, _ vecx: [Double], _ incx: Int, _ vecy: [Double], _ incy: Int, _ matA: inout [Double], _ lda: Int)
    {
        metalXsyr2(order, uplo, N, alpha, vecx, incx, vecy, incy, &matA, lda)
    }
}
