//
//  metalRotmg.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

import Metal

public extension MetalBlas
{
    private func metalXrotmg<T: BinaryFloatingPoint>(_ d1: inout T, _ d2: inout T, _ b1: inout T, _ b2: T, _ P: inout [T])
    {
        // No benefit from GPU, just do on CPU

        // from netlib reference
        // netlib has 4096 for float; float16 only has 5 exponent bits
        let gam : T = T.self == Float16.self ? 128 : 4096
        let gamsq : T = gam * gam
        let rgamsq = 1.0 / gamsq

        var sflag : T = -1
        var sh11 : T = 0
        var sh12 : T = 0
        var sh21 : T = 0
        var sh22 : T = 0
        if d1 < 0
        {
            // set params at end
            d1 = 0
            d2 = 0
            b1 = 0
        }
        else
        {
            let p2 = d2 * b2
            if p2 == 0
            {
                P[0] = -2
                return
            }

            let p1 = d1 * b1
            let q2 = p2 * b2
            let q1 = p1 * b1
            var su : T = 0

            if abs(q1) > abs(q2)
            {
                sh21 = -b2 / b1
                sh12 = p2 / p1

                su = 1 - sh12 * sh21
                if(su > 0)
                {
                    sflag = 0
                    d1 = d1 / su
                    d2 = d2 / su
                    b1 = b1 * su
                }
            }
            else
            {
                if q2 < 0
                {
                    sflag = -1
                    sh11 = 0; sh12 = 0; sh21 = 0; sh22 = 0
                    d1 = 0; d2 = 0; b1 = 0
                }
                else
                {
                    sflag = 1
                    sh11 = p1 / p2
                    sh22 = b1 / b2
                    su = 1.0 + sh11 * sh22
                    let tmp : T = d2 / su
                    d2 = d1 / su
                    d1 = tmp
                    b1 = b2 * su
                }
            }

            if d1 != 0
            {
                while d1 <= rgamsq || d1 >= gamsq
                {
                    if sflag == 0
                    {
                        sh11 = 1; sh22 = 1; sflag = -1
                    }
                    else
                    {
                        sh21 = -1; sh12 = 1; sflag = -1
                    }

                    if d1 <= rgamsq
                    {
                        d1 = d1 * gamsq
                        b1 = b1 / gam
                        sh11 = sh11 / gam
                        sh12 = sh12 / gam
                    }
                    else
                    {
                        d1 = d1 / gamsq
                        b1 = b1 * gam
                        sh11 = sh11 * gam
                        sh12 = sh12 * gam
                    }
                }
            }
        }

        if sflag < 0
        {
            P[1] = sh11
            P[2] = sh21
            P[3] = sh12
            P[4] = sh22
        }
        else if sflag == 0
        {
            P[2] = sh21
            P[3] = sh12
        }
        else
        {
            P[1] = sh11
            P[4] = sh22
        }

        P[0] = sflag
        
    }

    private func metalXrotmg<T: BinaryFloatingPoint>(_ type: T, _ bufd1: inout MTLBuffer, _ bufd2: inout MTLBuffer, _ bufb1: inout MTLBuffer, _ bufb2: MTLBuffer, _ bufP: inout MTLBuffer)
    {
        assert(T.self == Float.self || T.self == Float16.self)

        // Encode commands
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create command buffer or compute encoder.")
        }

        let funcString = T.self == Float.self ? "srotmg" : "hrotmg"
        guard let pipelineState = pipelineStates[funcString] else {
            fatalError("Failed to get pipeline state for function rotmg")
        }
        computeEncoder.setComputePipelineState(pipelineState)

        computeEncoder.setBuffer(bufd1, offset: 0, index: 0)
        computeEncoder.setBuffer(bufd2, offset: 0, index: 1)
        computeEncoder.setBuffer(bufb1, offset: 0, index: 2)
        computeEncoder.setBuffer(bufb2, offset: 0, index: 3)
        computeEncoder.setBuffer(bufP, offset: 0, index: 4)

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

    func metalHrotmg(_ bufd1: inout MTLBuffer, _ bufd2: inout MTLBuffer, _ bufb1: inout MTLBuffer, _ bufb2: MTLBuffer, _ bufP: inout MTLBuffer)
    {
        metalXrotmg(Float16(0), &bufd1, &bufd2, &bufb1, bufb2, &bufP)
    }

    func metalSrotmg(_ bufd1: inout MTLBuffer, _ bufd2: inout MTLBuffer, _ bufb1: inout MTLBuffer, _ bufb2: MTLBuffer, _ bufP: inout MTLBuffer)
    {
        metalXrotmg(Float(0), &bufd1, &bufd2, &bufb1, bufb2, &bufP)
    }

    func metalDrotmg(_ bufd1: inout MTLBuffer, _ bufd2: inout MTLBuffer, _ bufb1: inout MTLBuffer, _ bufb2: MTLBuffer, _ bufP: inout MTLBuffer)
    {
        metalXrotmg(Double(0), &bufd1, &bufd2, &bufb1, bufb2, &bufP)
    }

    func metalHrotmg(_ d1: inout Float16, _ d2: inout Float16, _ b1: inout Float16, _ b2: Float16, _ P: inout [Float16])
    {
        metalXrotmg(&d1, &d2, &b1, b2, &P)
    }

    func metalSrotmg(_ d1: inout Float, _ d2: inout Float, _ b1: inout Float, _ b2: Float, _ P: inout [Float])
    {
        metalXrotmg(&d1, &d2, &b1, b2, &P)
    }

    func metalDrotmg(_ d1: inout Double, _ d2: inout Double, _ b1: inout Double, _ b2: Double, _ P: inout [Double])
    {
        metalXrotmg(&d1, &d2, &b1, b2, &P)
    }
}
