//
//  metalRot.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-25.
//

import Metal

public extension MetalBlas
{
    private func metalXrot<T: BinaryFloatingPoint>(_ N: Int, _ vecx: inout [T], _ incx: Int, _ vecy: inout [T], _ incy: Int, _ c: T, _ s: T)
    {
        // TODO: potentially have smarter memory handling in this class
        // e.x. on MetalBlas init, create buffer with private access with X bytes, and if the buffer has enough, just use that buffer here
        guard
            var bufferx = device.makeBuffer(bytes: vecx, length: MemoryLayout<T>.stride * N * incx, options: [.storageModeManaged]),
            var buffery = device.makeBuffer(bytes: vecy, length: MemoryLayout<T>.stride * N * incy, options: [.storageModeShared])//[.storageModeManaged])
        else {
            fatalError("Failed to create buffers for matrices in rot")
        }

        metalXrot(N, &bufferx, incx, &buffery, incy, c, s)

        // Copy the data into the array
        copyBufToArray(bufferx, &vecx)
        copyBufToArray(buffery, &vecy)
    }

    private func metalXrot<T: BinaryFloatingPoint>(_ N: Int, _ bufferx: inout MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int, _ c: T, _ s: T)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcString = T.self == Float.self ? "srot" : "hrot"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function rot")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        var Nm = N, incxm = incx, incym = incy
        var Cm = c, Sm = s
        let bufferN = device.makeBuffer(bytes: &Nm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncx = device.makeBuffer(bytes: &incxm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncy = device.makeBuffer(bytes: &incym, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferC = device.makeBuffer(bytes: &Cm, length: MemoryLayout<T>.stride, options: .storageModePrivate)
        let bufferS = device.makeBuffer(bytes: &Sm, length: MemoryLayout<T>.stride, options: .storageModePrivate)

        computeEncoder.setBuffer(bufferN, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferx, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferIncx, offset: 0, index: 2)
        computeEncoder.setBuffer(buffery, offset: 0, index: 3)
        computeEncoder.setBuffer(bufferIncy, offset: 0, index: 4)
        computeEncoder.setBuffer(bufferC, offset: 0, index: 5)
        computeEncoder.setBuffer(bufferS, offset: 0, index: 6)

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

    func metalHrot(_ N: Int, _ bufferx: inout MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int, _ c: Float16, _ s: Float16)
    {
        metalXrot(N, &bufferx, incx, &buffery, incy, c, s)
    }

    func metalSrot(_ N: Int, _ bufferx: inout MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int, _ c: Float, _ s: Float)
    {
        metalXrot(N, &bufferx, incx, &buffery, incy, c, s)
    }

    func metalDrot(_ N: Int, _ bufferx: inout MTLBuffer, _ incx: Int, _ buffery: inout MTLBuffer, _ incy: Int, _ c: Double, _ s: Double)
    {
        metalXrot(N, &bufferx, incx, &buffery, incy, c, s)
    }

    func metalHrot(_ N: Int, _ vecx: inout [Float16], _ incx: Int, _ vecy: inout [Float16], _ incy: Int, _ c: Float16, _ s: Float16)
    {
        metalXrot(N, &vecx, incx, &vecy, incy, c, s)
    }

    func metalSrot(_ N: Int, _ vecx: inout [Float], _ incx: Int, _ vecy: inout [Float], _ incy: Int, _ c: Float, _ s: Float)
    {
        metalXrot(N, &vecx, incx, &vecy, incy, c, s)
    }

    func metalDrot(_ N: Int, _ vecx: inout [Double], _ incx: Int, _ vecy: inout [Double], _ incy: Int, _ c: Double, _ s: Double)
    {
        metalXrot(N, &vecx, incx, &vecy, incy, c, s)
    }
}
