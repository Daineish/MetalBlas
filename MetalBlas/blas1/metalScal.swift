//
//  metalScal.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Metal

public extension MetalBlas
{
    func metalXscal<T: BinaryFloatingPoint>(_ N: Int, _ alpha: T, _ vecx: inout [T], _ incx: Int)
    {
        // TODO: potentially have smarter memory handling in this class
        guard
            var bufferx = device.makeBuffer(bytes: vecx, length: MemoryLayout<T>.stride * N * incx, options: .storageModeShared)
        else {
            fatalError("Failed to create buffers for matrices in sscal")
        }
        metalXscal(N, alpha, &bufferx, incx)

        // Copy the data into the array
        copyBufToArray(bufferx, &vecx)
    }

    func metalXscal<T: BinaryFloatingPoint>(_ N: Int, _ alpha: T, _ bufferx: inout MTLBuffer, _ incx: Int)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }
        
        let funcString = T.self == Float.self ? "sscal" : "hscal"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function scal")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        var Nm = N, alpham = alpha, incxm = incx
        let bufferN = device.makeBuffer(bytes: &Nm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncx = device.makeBuffer(bytes: &incxm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferAlpha = device.makeBuffer(bytes: &alpham, length: MemoryLayout<T>.stride, options: .storageModePrivate)

        computeEncoder.setBuffer(bufferN, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferAlpha, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferx, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferIncx, offset: 0, index: 3)

        let w = pipelineState.threadExecutionWidth
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

    func metalHscal(_ N: Int, _ alpha: Float16, _ bufx: inout MTLBuffer, _ incx: Int)
    {
        metalXscal(N, alpha, &bufx, incx)
    }

    func metalHscal(_ N: Int, _ alpha: Float16, _ vecx: inout [Float16], _ incx: Int)
    {
        metalXscal(N, alpha, &vecx, incx)
    }

    func metalSscal(_ N: Int, _ alpha: Float, _ bufx: inout MTLBuffer, _ incx: Int)
    {
        metalXscal(N, alpha, &bufx, incx)
    }

    func metalSscal(_ N: Int, _ alpha: Float, _ vecx: inout [Float], _ incx: Int)
    {
        metalXscal(N, alpha, &vecx, incx)
    }

    func metalDscal(_ N: Int, _ alpha: Double, _ bufx: inout MTLBuffer, _ incx: Int)
    {
        metalXscal(N, alpha, &bufx, incx)
    }

    func metalDscal(_ N: Int, _ alpha: Double, _ vecx: inout [Double], _ incx: Int)
    {
        metalXscal(N, alpha, &vecx, incx)
    }
}

