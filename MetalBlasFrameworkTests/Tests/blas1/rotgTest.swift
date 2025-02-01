//
//  rotgTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-29.
//

import XCTest
@testable import MetalBlasFramework

class RotgFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var aH, bH, cH, sH : Float16!
    var aF, bF, cF, sF : Float!
    var aD, bD, cD, sD : Double!

    var aBuf : MTLBuffer!
    var bBuf : MTLBuffer!
    var cBuf : MTLBuffer!
    var sBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        metalBlas = metalBlasIn!

        if T.self == Float.self
        {
            aF = params.alpha
            bF = params.beta
            cF = -1
            sF = -1

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(&aF, [.storageModeManaged])
                bBuf = metalBlas.getDeviceBuffer(&bF, [.storageModeManaged])
                cBuf = metalBlas.getDeviceBuffer(&cF, [.storageModeManaged])
                sBuf = metalBlas.getDeviceBuffer(&sF, [.storageModeManaged])
            }
        }
        else if T.self == Double.self
        {
            aD = Double(params.alpha)
            bD = Double(params.beta)
            cD = -1
            sD = -1

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(&aD, [.storageModeManaged])
                bBuf = metalBlas.getDeviceBuffer(&bD, [.storageModeManaged])
                cBuf = metalBlas.getDeviceBuffer(&cD, [.storageModeManaged])
                sBuf = metalBlas.getDeviceBuffer(&sD, [.storageModeManaged])
            }
        }
        else if T.self == Float16.self
        {
            aH = Float16(params.alpha)
            bH = Float16(params.beta)
            cH = -1
            sH  = -1

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(&aH, [.storageModeManaged])
                bBuf = metalBlas.getDeviceBuffer(&bH, [.storageModeManaged])
                cBuf = metalBlas.getDeviceBuffer(&cH, [.storageModeManaged])
                sBuf = metalBlas.getDeviceBuffer(&sH, [.storageModeManaged])
            }
        }
    }

    private func callRotg(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSrotg(&aF, &bF, &cF, &sF) :
            T.self == Double.self ? refDrotg(&aD, &bD, &cD, &sD) :
                                    refHrotg(&aH, &bH, &cH, &sH)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSrotg(&aBuf, &bBuf, &cBuf, &sBuf) :
            T.self == Double.self ? metalBlas.metalDrotg(&aBuf, &bBuf, &cBuf, &sBuf) :
                                    metalBlas.metalHrotg(&aBuf, &bBuf, &cBuf, &sBuf)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSrotg(&aF, &bF, &cF, &sF) :
            T.self == Double.self ? metalBlas.metalDrotg(&aD, &bD, &cD, &sD) :
                                    metalBlas.metalHrotg(&aH, &bH, &cH, &sH)
        }
    }

    func validateRotg() -> Bool
    {
        if T.self == Float.self
        {
            var aCpy = aF!
            var bCpy = bF!
            var cCpy = cF!
            var sCpy = sF!
            refSrotg(&aCpy, &bCpy, &cCpy, &sCpy)
            callRotg(false)
            if useBuffers
            {
                metalBlas.copyBufToVal(aBuf, &aF)
                metalBlas.copyBufToVal(bBuf, &bF)
                metalBlas.copyBufToVal(cBuf, &cF)
                metalBlas.copyBufToVal(sBuf, &sF)
            }

            return printIfNotEqual(aF, aCpy) && printIfNotEqual(bF, bCpy) && printIfNotEqual(cF, cCpy) && printIfNotEqual(sF, sCpy)
        }
        else if T.self == Double.self
        {
            var aCpy = aD!
            var bCpy = bD!
            var cCpy = cD!
            var sCpy = sD!
            refDrotg(&aCpy, &bCpy, &cCpy, &sCpy)
            callRotg(false)
            if useBuffers
            {
                metalBlas.copyBufToVal(aBuf, &aD)
                metalBlas.copyBufToVal(bBuf, &bD)
                metalBlas.copyBufToVal(cBuf, &cD)
                metalBlas.copyBufToVal(sBuf, &sD)
            }

            return printIfNotEqual(aD, aCpy) && printIfNotEqual(bD, bCpy) && printIfNotEqual(cD, cCpy) && printIfNotEqual(sD, sCpy)
        }
        else if T.self == Float16.self
        {
            var aCpy = aH!
            var bCpy = bH!
            var cCpy = cH!
            var sCpy = sH!
            refHrotg(&aCpy, &bCpy, &cCpy, &sCpy)
            callRotg(false)
            if useBuffers
            {
                metalBlas.copyBufToVal(aBuf, &aH)
                metalBlas.copyBufToVal(bBuf, &bH)
                metalBlas.copyBufToVal(cBuf, &cH)
                metalBlas.copyBufToVal(sBuf, &sH)
            }

            return printIfNotEqual(aH, aCpy) && printIfNotEqual(bH, bCpy) && printIfNotEqual(cH, cCpy) && printIfNotEqual(sH, sCpy)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkRotg(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callRotg(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callRotg(benchRef)
                }
            }
        )
        return result
    }
}

class rotgTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/rotgInput"
    let metalBlas = MetalBlas()
    var useBuffersDirectly = false
    var paramLineNum = 0

    override func setUp()
    {
        super.setUp()
    }

    override func tearDown()
    {
        super.tearDown()
    }

    func setupParams(_ paramLineNum: Int)
    {
        params = getParams(fileName: fileName, i: paramLineNum)
        useBuffersDirectly = params.useBuffers
    }

    func printTestInfo(_ name: String)
    {
        print(name)
        print("\tfunc,prec,N,a,b,useBuf,coldIters,hotIters")
        print("\trotg", params.prec, params.alpha, params.beta, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func rotgTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let rotgFramework = RotgFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = rotgFramework.validateRotg()
            XCTAssert(pass)
        }

        var result = rotgFramework.benchmarkRotg(false, params.coldIters, params.hotIters)
        var accResult = rotgFramework.benchmarkRotg(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getRotgGflopCount() / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getRotgGflopCount() / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func rotgLauncher()
    {
        if params.prec == precisionType.fp32
        {
            rotgTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            rotgTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            rotgTester(Float16(0))
        }
    }

    func testRotg0()
    {
        for i in 0...7
        {
            setupParams(i)
            printTestInfo("testRotg" + String(i))
            rotgLauncher()
        }
    }
}

