//
//  rotmTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-26.
//

import XCTest
@testable import MetalBlasFramework

class RotmFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var yArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var yArrF : [ Float ]!
    var xArrD : [ Double ]!
    var yArrD : [ Double ]!
    var pH : [ Float16 ]!
    var pF : [ Float ]!
    var pD : [ Double ]!
    let N : Int
    let incx : Int
    let incy : Int

    var xBuf : MTLBuffer!
    var yBuf : MTLBuffer!
    var pBuf : MTLBuffer!

    let flag : Float

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        N = params.N
        incx = params.incx
        incy = params.incy
        // TODO: P init
        flag = params.alpha

        metalBlas = metalBlasIn!

        let sizeX : Int = params.N * params.incx
        let sizeY : Int = params.N * params.incy
        let sizeP : Int = 5

        if T.self == Float.self
        {
            xArrF = []; yArrF = []; pF = []
            initRandom(&xArrF, sizeX)
            initRandom(&yArrF, sizeY)
            initRandom(&pF, sizeP)
            pF[0] = flag
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrF, M: yArrF.count, [.storageModeManaged])!// .storageModeManaged)!
                pBuf = metalBlas.getDeviceBuffer(matA: pF, M: pF.count, [.storageModeManaged])
            }
        }
        else if T.self == Double.self
        {
            xArrD = []; yArrD = []; pD = []
            initRandom(&xArrD, sizeX)
            initRandom(&yArrD, sizeY)
            initRandom(&pD, sizeP)
            pD[0] = Double(flag)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrD, M: yArrD.count, [.storageModeManaged])!// .storageModeManaged)!
                pBuf = metalBlas.getDeviceBuffer(matA: pD, M: pD.count, [.storageModeManaged])
            }
        }
        else if T.self == Float16.self
        {
            xArrH = []; yArrH = []; pH = [0,0,0,0,0]
            initRandom(&xArrH, sizeX, (1...5))
            initRandom(&yArrH, sizeY, (1...5))
            // TODO: deal with precision issues
            var pI : [Int] = [0,0,0,0,0]
            initRandom(&pI, sizeP, (1...10))
            pH[1] = Float16(pI[1])
            pH[2] = Float16(pI[2])
            pH[3] = Float16(pI[3])
            pH[4] = Float16(pI[4])
            pH[0] = Float16(flag)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrH, M: yArrH.count, [.storageModeManaged])!// .storageModeManaged)!
                pBuf = metalBlas.getDeviceBuffer(matA: pH, M: pH.count, [.storageModeManaged])
            }
        }
    }

    private func callRotm(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSrotm(N, &xArrF, incx, &yArrF, incy, pF) :
            T.self == Double.self ? refDrotm(N, &xArrD, incx, &yArrD, incy, pD) :
                                    refHrotm(N, &xArrH, incx, &yArrH, incy, pH)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSrotm(N, &xBuf, incx, &yBuf, incy, pBuf) :
            T.self == Double.self ? metalBlas.metalDrotm(N, &xBuf, incx, &yBuf, incy, pBuf) :
                                    metalBlas.metalHrotm(N, &xBuf, incx, &yBuf, incy, pBuf)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSrotm(N, &xArrF, incx, &yArrF, incy, pF) :
            T.self == Double.self ? metalBlas.metalDrotm(N, &xArrD, incx, &yArrD, incy, pD) :
                                    metalBlas.metalHrotm(N, &xArrH, incx, &yArrH, incy, pH)
        }
    }

    func validateRotm() -> Bool
    {
        if T.self == Float.self
        {
            var xCpy = xArrF!
            var yCpy = yArrF!
            refSrotm(N, &xCpy, incx, &yCpy, incy, pF)
            callRotm(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrF)
                metalBlas.copyBufToArray(yBuf, &yArrF)
            }

            return printIfNotEqual(yArrF, yCpy) && printIfNotEqual(xArrF, xCpy)
        }
        else if T.self == Double.self
        {
            var xCpy = xArrD!
            var yCpy = yArrD!
            refDrotm(N, &xCpy, incx, &yCpy, incy,pD)
            callRotm(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrD)
                metalBlas.copyBufToArray(yBuf, &yArrD)
            }

            return printIfNotEqual(yArrD, yCpy) && printIfNotEqual(xArrD, xCpy)
        }
        else if T.self == Float16.self
        {
            var xCpy = xArrH!
            var yCpy = yArrH!
            refHrotm(N, &xCpy, incx, &yCpy, incy, pH)
            callRotm(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrH)
                metalBlas.copyBufToArray(yBuf, &yArrH)
            }

            return printIfNotNear(yArrH, yCpy, N) && printIfNotNear(xArrH, xCpy, N)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkRotm(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callRotm(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callRotm(benchRef)
                }
            }
        )
        return result
    }
}

class rotmTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/rotmInput"
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
        print("\tfunc,prec,N,incx,incy,flag,useBuf,coldIters,hotIters")
        print("\trotm", params.prec, params.N, params.incx, params.incy, params.alpha,params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func rotmTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let rotmFramework = RotmFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = rotmFramework.validateRotm()
            XCTAssert(pass)
        }

        var result = rotmFramework.benchmarkRotm(false, params.coldIters, params.hotIters)
        var accResult = rotmFramework.benchmarkRotm(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getRotmGflopCount(N: params.N, flag: params.alpha) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getRotmGflopCount(N: params.N, flag: params.alpha) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func rotmLauncher()
    {
        if params.prec == precisionType.fp32
        {
            rotmTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            rotmTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            rotmTester(Float16(0))
        }
    }

    func testRotm0()
    {
        // TODO: test more than one line per test? Probably on a per-file basis or something
        for i in 0...15
        {
            setupParams(i)
            printTestInfo("testRotm" + String(i))
            rotmLauncher()
        }
//        setupParams(0)
//        printTestInfo("testRotm0")
//        rotmLauncher()
    }

//    func testRotm1()
//    {
//        setupParams(1)
//        printTestInfo("testRotm1")
//        rotmLauncher()
//    }
//
//    func testRotm2()
//    {
//        setupParams(2)
//        printTestInfo("testRotm2")
//        rotmLauncher()
//    }
//
//    func testRotm3()
//    {
//        setupParams(3)
//        printTestInfo("testRotm3")
//        rotmLauncher()
//    }
}

