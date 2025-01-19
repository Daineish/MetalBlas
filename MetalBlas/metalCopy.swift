//
//  metalCopy.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

import Metal

public extension MetalBlas
{
    private func metalXcopy<T: BinaryFloatingPoint>(_ N: Int, _ vecx: [T], _ incx: Int, _ vecy: inout [T], _ incy: Int)
    {
        // TODO: potentially have smarter memory handling in this class
        // e.x. on MetalBlas init, create buffer with private access with X bytes, and if the buffer has enough, just use that buffer here
        guard
            let bufferx = device.makeBuffer(bytes: vecx, length: MemoryLayout<T>.stride * N * incx, options: [.storageModeManaged]),
            var buffery = device.makeBuffer(bytes: vecy, length: MemoryLayout<T>.stride * N * incy, options: [.storageModeShared])//[.storageModeManaged])
        else {
            fatalError("Failed to create buffers for matrices in metalCopy")
        }

        metalXcopy(T(0), N, bufferx, incx, &buffery, incy)

        // Copy the data into the array
        copyBufToArray(buffery, &vecy)
    }

    private func metalXcopy<T: BinaryFloatingPoint>(_ type: T, _ N: Int, _ bufferx: MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcString = T.self == Float.self ? "scopy" : "hcopy"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function copy=")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        var Nm = N, incxm = incx, incym = incy
        let bufferN = device.makeBuffer(bytes: &Nm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncx = device.makeBuffer(bytes: &incxm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncy = device.makeBuffer(bytes: &incym, length: MemoryLayout<Int>.stride, options: .storageModePrivate)

        computeEncoder.setBuffer(bufferN, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferx, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferIncx, offset: 0, index: 2)
        computeEncoder.setBuffer(buffery, offset: 0, index: 3)
        computeEncoder.setBuffer(bufferIncy, offset: 0, index: 4)

        let w = pipelineState.threadExecutionWidth
        let threadsPerThreadgroup = MTLSizeMake(w, 1, 1)

        let threadgroups = MTLSize(
            width: N / w + 1,
            height: 1,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

//        // for managed buffer API
//        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
//            fatalError("Faled to create blit encoder.")
//        }
//
//        blitEncoder.synchronize(resource: buffery)
//        blitEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func metalHcopy(_ N: Int, _ bufferx: MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int)
    {
        metalXcopy(Float16(0), N, bufferx, incx, &buffery, incy)
    }

    func metalScopy(_ N: Int, _ bufferx: MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int)
    {
        metalXcopy(Float(0), N, bufferx, incx, &buffery, incy)
    }

    func metalDcopy(_ N: Int, _ bufferx: MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int)
    {
        metalXcopy(Double(0), N, bufferx, incx, &buffery, incy)
    }

    func metalHcopy(_ N: Int, _ vecx: [Float16], _ incx: Int, _ vecy: inout [Float16], _ incy: Int)
    {
        metalXcopy(N, vecx, incx, &vecy, incy)
    }

    func metalScopy(_ N: Int, _ vecx: [Float], _ incx: Int, _ vecy: inout [Float], _ incy: Int)
    {
        metalXcopy(N, vecx, incx, &vecy, incy)
    }

    func metalDcopy(_ N: Int, _ vecx: [Double], _ incx: Int, _ vecy: inout [Double], _ incy: Int)
    {
        metalXcopy(N, vecx, incx, &vecy, incy)
    }
}
