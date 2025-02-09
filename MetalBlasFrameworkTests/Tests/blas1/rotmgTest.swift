//
//  rotmgTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

import XCTest
@testable import MetalBlasFramework

class RotmgFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var d1H, d2H, b1H, b2H : Float16!
    var d1F, d2F, b1F, b2F : Float!
    var d1D, d2D, b1D, b2D : Double!
    var pH : [ Float16 ]!
    var pF : [ Float ]!
    var pD : [ Double ]!

    var d1Buf : MTLBuffer!
    var d2Buf : MTLBuffer!
    var b1Buf : MTLBuffer!
    var b2Buf : MTLBuffer!
    var pBuf : MTLBuffer!

    let flag : Float

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        metalBlas = metalBlasIn!
        let sizeP : Int = 5

        // TODO: smarter initialization
        flag = params.alpha

        if T.self == Float.self
        {
            pF = []
            d1F = 0; d2F = 0; b1F = 0; b2F = 0
            initRandom(&d1F); initRandom(&d2F)
            initRandom(&b1F); initRandom(&b2F)
            initRandom(&pF, sizeP)
            pF[0] = flag
            if useBuffers
            {
                d1Buf = metalBlas.getDeviceBuffer(&d1F, [.storageModeManaged])
                d2Buf = metalBlas.getDeviceBuffer(&d2F, [.storageModeManaged])
                b1Buf = metalBlas.getDeviceBuffer(&b1F, [.storageModeManaged])
                b2Buf = metalBlas.getDeviceBuffer(&b2F, [.storageModeManaged])
                pBuf = metalBlas.getDeviceBuffer(matA: pF, M: pF.count, [.storageModeManaged])
            }
        }
        else if T.self == Double.self
        {
            pD = []
            d1D = 0; d2D = 0; b1D = 0; b2D = 0
            initRandom(&d1D); initRandom(&d2D)
            initRandom(&b1D); initRandom(&b2D)
            initRandom(&pD, sizeP)
            pD[0] = Double(flag)

            if useBuffers
            {
                d1Buf = metalBlas.getDeviceBuffer(&d1D, [.storageModeManaged])
                d2Buf = metalBlas.getDeviceBuffer(&d2D, [.storageModeManaged])
                b1Buf = metalBlas.getDeviceBuffer(&b1D, [.storageModeManaged])
                b2Buf = metalBlas.getDeviceBuffer(&b2D, [.storageModeManaged])
                pBuf = metalBlas.getDeviceBuffer(matA: pD, M: pD.count, [.storageModeManaged])
            }
        }
        else if T.self == Float16.self
        {
            pH = []
            d1H = 0; d2H = 0; b1H = 0; b2H = 0
            initRandom(&d1H); initRandom(&d2H)
            initRandom(&b1H); initRandom(&b2H)
            initRandom(&pH, sizeP)
            pH[0] = Float16(flag)

            if useBuffers
            {
                d1Buf = metalBlas.getDeviceBuffer(&d1H, [.storageModeManaged])
                d2Buf = metalBlas.getDeviceBuffer(&d2H, [.storageModeManaged])
                b1Buf = metalBlas.getDeviceBuffer(&b1H, [.storageModeManaged])
                b2Buf = metalBlas.getDeviceBuffer(&b2H, [.storageModeManaged])
                pBuf = metalBlas.getDeviceBuffer(matA: pH, M: pH.count, [.storageModeManaged])
            }
        }
    }

    private func callRotmg(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSrotmg(&d1F, &d2F, &b1F, b2F, &pF) :
            T.self == Double.self ? refDrotmg(&d1D, &d2D, &b1D, b2D, &pD) :
                                    refHrotmg(&d1H, &d2H, &b1H, b2H, &pH)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSrotmg(&d1Buf, &d2Buf, &b1Buf, b2Buf, &pBuf) :
            T.self == Double.self ? metalBlas.metalDrotmg(&d1Buf, &d2Buf, &b1Buf, b2Buf, &pBuf) :
                                    metalBlas.metalHrotmg(&d1Buf, &d2Buf, &b1Buf, b2Buf, &pBuf)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSrotmg(&d1F, &d2F, &b1F, b2F, &pF) :
            T.self == Double.self ? metalBlas.metalDrotmg(&d1D, &d2D, &b1D, b2D, &pD) :
                                    metalBlas.metalHrotmg(&d1H, &d2H, &b1H, b2H, &pH)
        }
    }

    func validateRotmg() -> Bool
    {
        if T.self == Float.self
        {
            var d1Cpy = d1F!
            var d2Cpy = d2F!
            var b1Cpy = b1F!
            var pCpy = pF!
            refSrotmg(&d1Cpy, &d2Cpy, &b1Cpy, b2F, &pCpy)
            callRotmg(false)
            if useBuffers
            {
                metalBlas.copyBufToVal(d1Buf, &d1F)
                metalBlas.copyBufToVal(d2Buf, &d2F)
                metalBlas.copyBufToVal(b1Buf, &b1F)
                metalBlas.copyBufToVal(b2Buf, &b2F)
                metalBlas.copyBufToArray(pBuf, &pF)
            }

            return printIfNotNear(d1F, d1Cpy, 100, true, "d1") && printIfNotNear(d2F, d2Cpy, 100, true, "d2") && printIfNotNear(b1F, b1Cpy, 100, true, "b1") && printIfNotNear(pF, pCpy, 100, true, "parr")
        }
        else if T.self == Double.self
        {
            var d1Cpy = d1D!
            var d2Cpy = d2D!
            var b1Cpy = b1D!
            var pCpy = pD!
            refDrotmg(&d1Cpy, &d2Cpy, &b1Cpy, b2D, &pCpy)
            callRotmg(false)
            if useBuffers
            {
                metalBlas.copyBufToVal(d1Buf, &d1D)
                metalBlas.copyBufToVal(d2Buf, &d2D)
                metalBlas.copyBufToVal(b1Buf, &b1D)
                metalBlas.copyBufToVal(b2Buf, &b2D)
                metalBlas.copyBufToArray(pBuf, &pD)
            }

            return printIfNotNear(d1D, d1Cpy, 100, true, "d1") && printIfNotNear(d2D, d2Cpy, 100, true, "d2") && printIfNotNear(b1D, b1Cpy, 100, true, "b1") && printIfNotNear(pD, pCpy, 100, true, "parr")
        }
        else if T.self == Float16.self
        {
            var d1Cpy = d1H!
            var d2Cpy = d2H!
            var b1Cpy = b1H!
            var pCpy = pH!
            refHrotmg(&d1Cpy, &d2Cpy, &b1Cpy, b2H, &pCpy)
            callRotmg(false)
            if useBuffers
            {
                metalBlas.copyBufToVal(d1Buf, &d1H)
                metalBlas.copyBufToVal(d2Buf, &d2H)
                metalBlas.copyBufToVal(b1Buf, &b1H)
                metalBlas.copyBufToVal(b2Buf, &b2H)
                metalBlas.copyBufToArray(pBuf, &pH)
            }

            return printIfNotNear(d1H, d1Cpy, 100, true, "d1") && printIfNotNear(d2H, d2Cpy, 100, true, "d2") && printIfNotNear(b1H, b1Cpy, 100, true, "b1") && printIfNotNear(pH, pCpy, 100, true, "parr")
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkRotmg(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callRotmg(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callRotmg(benchRef)
                }
            }
        )
        return result
    }
}

class rotmgTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas1/rotmgInput"
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
        print("\tfunc,prec,flag,useBuf,coldIters,hotIters")
        print("\trotmg", params.prec, params.alpha, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func rotmgTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let rotmgFramework = RotmgFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = rotmgFramework.validateRotmg()
            XCTAssert(pass)
        }

        var result = rotmgFramework.benchmarkRotmg(false, params.coldIters, params.hotIters)
        var accResult = rotmgFramework.benchmarkRotmg(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getRotmgGflopCount() / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getRotmgGflopCount() / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func rotmgLauncher()
    {
        if params.prec == precisionType.fp32
        {
            rotmgTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            rotmgTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            rotmgTester(Float16(0))
        }
    }

    func testRotmg0()
    {
        for i in 0...11
        {
            setupParams(i)
            printTestInfo("testRotmg" + String(i))
            for _ in 0..<100
            {
                // TODO: get better code coverage than just hoping
                rotmgLauncher()
            }
        }
    }
}

