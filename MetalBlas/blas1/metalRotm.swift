//
//  metalRotm.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-26.
//

import Metal

public extension MetalBlas
{
    private func metalXrotm<T: BinaryFloatingPoint>(_ N: Int, _ vecx: inout [T], _ incx: Int, _ vecy: inout [T], _ incy: Int, _ P: [T])
    {
        // TODO: potentially have smarter memory handling in this class
        // e.x. on MetalBlas init, create buffer with private access with X bytes, and if the buffer has enough, just use that buffer here
        guard
            var bufferx = device.makeBuffer(bytes: vecx, length: MemoryLayout<T>.stride * N * incx, options: [.storageModeManaged]),
            var buffery = device.makeBuffer(bytes: vecy, length: MemoryLayout<T>.stride * N * incy, options: [.storageModeShared]),//[.storageModeManaged])
            let bufferP = device.makeBuffer(bytes: P, length: MemoryLayout<T>.stride * 5, options: [.storageModePrivate])
        else {
            fatalError("Failed to create buffers for matrices in rotm")
        }

        metalXrotm(T(0), N, &bufferx, incx, &buffery, incy, bufferP)

        // Copy the data into the array
        copyBufToArray(bufferx, &vecx)
        copyBufToArray(buffery, &vecy)
    }

    private func metalXrotm<T: BinaryFloatingPoint>(_ type: T, _ N: Int, _ bufferx: inout MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int, _ bufferP: MTLBuffer)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcString = T.self == Float.self ? "srotm" : "hrotm"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function rotm")
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
        computeEncoder.setBuffer(bufferP, offset: 0, index: 5)

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

    func metalHrotm(_ N: Int, _ bufferx: inout MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int, _ bufferP: MTLBuffer)
    {
        metalXrotm(Float16(0), N, &bufferx, incx, &buffery, incy, bufferP)
    }

    func metalSrotm(_ N: Int, _ bufferx: inout MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int, _ bufferP: MTLBuffer)
    {
        metalXrotm(Float(0), N, &bufferx, incx, &buffery, incy, bufferP)
    }

    func metalDrotm(_ N: Int, _ bufferx: inout MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int, _ bufferP: MTLBuffer)
    {
        metalXrotm(Double(0), N, &bufferx, incx, &buffery, incy, bufferP)
    }

    func metalHrotm(_ N: Int, _ vecx: inout [Float16], _ incx: Int, _ vecy: inout [Float16], _ incy: Int, _ P: [Float16])
    {
        metalXrotm(N, &vecx, incx, &vecy, incy, P)
    }

    func metalSrotm(_ N: Int, _ vecx: inout [Float], _ incx: Int, _ vecy: inout [Float], _ incy: Int, _ P: [Float])
    {
        metalXrotm(N, &vecx, incx, &vecy, incy, P)
    }

    func metalDrotm(_ N: Int, _ vecx: inout [Double], _ incx: Int, _ vecy: inout [Double], _ incy: Int, _ P: [Double])
    {
        metalXrotm(N, &vecx, incx, &vecy, incy, P)
    }
}
