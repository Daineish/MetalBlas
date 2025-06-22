//
//  metalTpmv.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-06-21.
//

import Metal

public extension MetalBlas
{
    private func metalXtpmv<T: BinaryFloatingPoint>(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType, _ N: Int, _ matA: [T], _ vecx: inout [T], _ incx: Int)
    {
        let sizeA = (N * (N + 1)) / 2
        let sizex = N * incx
        guard
            let bufferA = device.makeBuffer(bytes: matA, length: MemoryLayout<T>.stride * sizeA, options: [.storageModeManaged]),
            var bufferx = device.makeBuffer(bytes: vecx, length: MemoryLayout<T>.stride * sizex, options: [.storageModeManaged])
        else {
            fatalError("Failed to create buffers for matrices in tpmv")
        }

        metalXtpmv(T(0), order, uplo, trans, diag, N, bufferA, &bufferx, incx)

        // Copy the data into the array
        copyBufToArray(bufferx, &vecx)
    }

    private func metalXtpmv<T: BinaryFloatingPoint>(_ type: T, _ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType, _ N: Int, _ bufferA: MTLBuffer, _ bufferx: inout MTLBuffer, _ incx: Int)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        var bufferWork = device.makeBuffer(length: MemoryLayout<T>.stride * N, options: .storageModeShared)!
        metalXcopy(T(0), N, bufferx, incx, &bufferWork, 1)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcStringTpmv = T.self == Float.self ? "stpmv" : "htpmv"
        guard let pipelineState = pipelineStates[funcStringTpmv] else {
            fatalError("Failed to get pipeline state for function tpmv")
        }

        computeEncoder.setComputePipelineState(pipelineState)

        var Nm = N, incxm = incx
        var orderm = order, uplom = uplo, transm = trans, diagm = diag
        let bufferN = device.makeBuffer(bytes: &Nm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncx = device.makeBuffer(bytes: &incxm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferOrder = device.makeBuffer(bytes: &orderm, length: MemoryLayout<OrderType>.stride, options: .storageModePrivate)
        let bufferUplo = device.makeBuffer(bytes: &uplom, length: MemoryLayout<UploType>.stride, options: .storageModePrivate)
        let bufferTrans = device.makeBuffer(bytes: &transm, length: MemoryLayout<TransposeType>.stride, options: .storageModePrivate)
        let bufferDiag = device.makeBuffer(bytes: &diagm, length: MemoryLayout<DiagType>.stride, options: .storageModePrivate)

        computeEncoder.setBuffer(bufferOrder, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferUplo, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferTrans, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferDiag, offset: 0, index: 3)
        computeEncoder.setBuffer(bufferN, offset: 0, index: 4)
        computeEncoder.setBuffer(bufferA, offset: 0, index: 5)
        computeEncoder.setBuffer(bufferx, offset: 0, index: 6)
        computeEncoder.setBuffer(bufferIncx, offset: 0, index: 7)
        computeEncoder.setBuffer(bufferWork, offset: 0, index: 8)

        let w = pipelineState.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)

        let threadgroups = MTLSize(width: N / w + 1, height: 1, depth: 1)

        computeEncoder.setComputePipelineState(pipelineState)

        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func metalHtpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType, _ N: Int, _ bufferA: MTLBuffer, _ bufferx: inout MTLBuffer, _ incx: Int)
    {
        metalXtpmv(Float16(0), order, uplo, trans, diag, N, bufferA, &bufferx, incx)
    }

    func metalStpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType, _ N: Int, _ bufferA: MTLBuffer, _ bufferx: inout MTLBuffer, _ incx: Int)
    {
        metalXtpmv(Float(0), order, uplo, trans, diag, N, bufferA, &bufferx, incx)
    }

    func metalDtpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType, _ N: Int,_ bufferA: MTLBuffer, _ bufferx: inout MTLBuffer, _ incx: Int)
    {
        metalXtpmv(Double(0), order, uplo, trans, diag, N, bufferA, &bufferx, incx)
    }

    func metalHtpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType, _ N: Int, _ matA: [Float16], _ vecx: inout [Float16], _ incx: Int)
    {
        metalXtpmv(order, uplo, trans, diag, N, matA, &vecx, incx)
    }

    func metalStpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType, _ N: Int, _ matA: [Float], _ vecx: inout [Float], _ incx: Int)
    {
        metalXtpmv(order, uplo, trans, diag, N, matA, &vecx, incx)
    }

    func metalDtpmv(_ order: OrderType, _ uplo: UploType, _ trans: TransposeType, _ diag: DiagType, _ N: Int, _ matA: [Double], _ vecx: inout [Double], _ incx: Int)
    {
        metalXtpmv(order, uplo, trans, diag, N, matA, &vecx, incx)
    }
}
