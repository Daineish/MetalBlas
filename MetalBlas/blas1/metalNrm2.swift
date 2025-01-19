//
//  metalNrm2.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

import Metal

public extension MetalBlas
{
    func metalXnrm2<T: BinaryFloatingPoint>(_ N: Int, _ vecx: [T], _ incx: Int) -> T
    {
        // TODO: potentially have smarter memory handling in this class
        var res = [T(0)]
        guard
            var bufferx = device.makeBuffer(bytes: vecx, length: MemoryLayout<T>.stride * N * incx, options: .storageModeShared),
            var bufferRes = device.makeBuffer(bytes: res, length: MemoryLayout<T>.stride, options: .storageModeShared) // .storageModePrivate
        else {
            fatalError("Failed to create buffers for matrices in asum")
        }
        metalXnrm2(T(0), N, bufferx, incx, &bufferRes)

        // Copy the data into the array
        copyBufToArray(bufferRes, &res)
        return res[0]
    }

    func metalXnrm2<T: BinaryFloatingPoint>(_ type: T, _ N: Int, _ bufferx: MTLBuffer, _ incx: Int, _ bufferRes: inout MTLBuffer)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }
        
        let funcString = T.self == Float.self ? "snrm2" : "hnrm2"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function nrm2")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        var Nm = N, incxm = incx
        let bufferN = device.makeBuffer(bytes: &Nm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        let bufferIncx = device.makeBuffer(bytes: &incxm, length: MemoryLayout<Int>.stride, options: .storageModePrivate)

        // TODO: Max like 2^25 elements
        let w = pipelineState.maxTotalThreadsPerThreadgroup
        let wSize = Int(ceil((Double(N) / 32) / Double(w)))
        var work : [T] = Array(repeating: 0, count: wSize)
        let bufferW = device.makeBuffer(bytes: &work, length: MemoryLayout<T>.stride * wSize, options: .storageModePrivate)

        computeEncoder.setBuffer(bufferN, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferx, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferIncx, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferRes, offset: 0, index: 3)
        computeEncoder.setBuffer(bufferW, offset: 0, index: 4)

        let threadsPerThreadgroup1 = MTLSizeMake(w, 1, 1)

        let threadgroups1 = MTLSize(
            width: wSize,
            height: 1,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadgroups1, threadsPerThreadgroup: threadsPerThreadgroup1)

        // single threadgroup reduction
        let tgfuncString = T.self == Float.self ? "sreducesq" : "hreducesq"
        guard let pipelineState2 = pipelineStates[tgfuncString] else {
            fatalError("Failed to get pipeline state for function reducesq")
        }
        computeEncoder.setComputePipelineState(pipelineState2)

        var wEls = wSize
        let bufferEls = device.makeBuffer(bytes: &wEls, length: MemoryLayout<Int>.stride, options: .storageModePrivate)
        
        computeEncoder.setBuffer(bufferEls, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferW, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferRes, offset: 0, index: 2)
        let threadgroups2 = MTLSize(width: wEls, height: 1, depth: 1)
        
        computeEncoder.dispatchThreads(threadgroups2, threadsPerThreadgroup: threadsPerThreadgroup1)

        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func metalHnrm2(_ N: Int, _ bufx: MTLBuffer, _ incx: Int, _ bufres: inout MTLBuffer)
    {
        metalXnrm2(Float16(0), N, bufx, incx, &bufres)
    }

    func metalHnrm2(_ N: Int, _ vecx: [Float16], _ incx: Int) -> Float16
    {
        return metalXnrm2(N, vecx, incx)
    }

    func metalSnrm2(_ N: Int, _ bufx: MTLBuffer, _ incx: Int, _ bufres: inout MTLBuffer)
    {
        metalXnrm2(Float(0), N, bufx, incx, &bufres)
    }

    func metalSnrm2(_ N: Int, _ vecx: [Float], _ incx: Int) -> Float
    {
        return metalXnrm2(N, vecx, incx)
    }

    func metalDnrm2(_ N: Int, _ bufx: MTLBuffer, _ incx: Int, _ bufres: inout MTLBuffer)
    {
        metalXnrm2(Double(0), N, bufx, incx, &bufres)
    }

    func metalDnrm2(_ N: Int, _ vecx: [Double], _ incx: Int) -> Double
    {
        return metalXnrm2(N, vecx, incx)
    }
}

