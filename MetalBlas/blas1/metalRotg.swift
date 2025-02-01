//
//  metalRotg.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-29.
//

import Metal

public extension MetalBlas
{
    private func metalXrotg<T: BinaryFloatingPoint>(_ a: inout T, _ b: inout T, _ c: inout T, _ s: inout T)
    {
        // No benefit from GPU, just do on CPU
        let scale = abs(a) + abs(b)
        if scale == 0.0
        {
            c = 1.0
            a = 0.0; b = 0.0; s = 0.0
        }
        else
        {
            let roe = abs(a) > abs(b) ? a : b
            let tmpA = a / scale
            let tmpB = b / scale
            var r = scale * sqrt(tmpA * tmpA + tmpB * tmpB)
            r = T(copysign(Float(r), Float(roe)))

            c = a / r
            s = b / r
            var z : T = 1.0
            if abs(a) > abs(b)
            {
                z = s
            }
            if abs(b) >= abs(a) && c != 0.0
            {
                z = 1.0 / c
            }

            a = r
            b = z
        }
    }

    private func metalXrotg<T: BinaryFloatingPoint>(_ type: T, _ bufA: inout MTLBuffer, _ bufB: inout MTLBuffer, _ bufC: inout MTLBuffer, _ bufS: inout MTLBuffer)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcString = T.self == Float.self ? "srotg" : "hrotg"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function rotg")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        computeEncoder.setBuffer(bufA, offset: 0, index: 0)
        computeEncoder.setBuffer(bufB, offset: 0, index: 1)
        computeEncoder.setBuffer(bufC, offset: 0, index: 2)
        computeEncoder.setBuffer(bufS, offset: 0, index: 3)

        let threadsPerThreadgroup = MTLSizeMake(1, 1, 1)

        let threadgroups = MTLSize(
            width: 1,
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

    func metalHrotg(_ bufA: inout MTLBuffer, _ bufB: inout MTLBuffer, _ bufC: inout MTLBuffer, _ bufS: inout MTLBuffer)
    {
        metalXrotg(Float16(0), &bufA, &bufB, &bufC, &bufS)
    }

    func metalSrotg(_ bufA: inout MTLBuffer, _ bufB: inout MTLBuffer, _ bufC: inout MTLBuffer, _ bufS: inout MTLBuffer)
    {
        metalXrotg(Float(0), &bufA, &bufB, &bufC, &bufS)
    }

    func metalDrotg(_ bufA: inout MTLBuffer, _ bufB: inout MTLBuffer, _ bufC: inout MTLBuffer, _ bufS: inout MTLBuffer)
    {
        metalXrotg(Double(0), &bufA, &bufB, &bufC, &bufS)
    }

    func metalHrotg(_ a: inout Float16, _ b: inout Float16, _ c: inout Float16, _ s: inout Float16)
    {
        metalXrotg(&a, &b, &c, &s)
    }

    func metalSrotg(_ a: inout Float, _ b: inout Float, _ c: inout Float, _ s: inout Float)
    {
        metalXrotg(&a, &b, &c, &s)
    }

    func metalDrotg(_ a: inout Double, _ b: inout Double, _ c: inout Double, _ s: inout Double)
    {
        metalXrotg(&a, &b, &c, &s)
    }
}
