//
//  dotTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

import XCTest
@testable import MetalBlasFramework

class DotFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var xArrD : [ Double ]!
    var yArrH : [ Float16 ]!
    var yArrF : [ Float ]!
    var yArrD : [ Double ]!
    var resH : [ Float16 ]!
    var resF : [ Float ]!
    var resD : [ Double ]!
    
    let N : Int
    let incx : Int
    let incy: Int

    var xBuf : MTLBuffer!
    var yBuf : MTLBuffer!
    var resBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        N = params.N
        incx = params.incx
        incy = params.incy

        metalBlas = metalBlasIn!

        let sizeX : Int = params.N * params.incx
        let sizeY : Int = params.N * params.incy

        if T.self == Float.self
        {
            xArrF = []
            yArrF = []
            resF = [0]
            
            initRandom(&xArrF, sizeX, (0...10))
            initRandom(&yArrF, sizeY, (0...10))

            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrF, M: yArrF.count, [.storageModeManaged])!
                resBuf = metalBlas.getDeviceBuffer(matA: resF, M: resF.count, [.storageModeManaged])
            }
        }
        else if T.self == Double.self
        {
            xArrD = []
            yArrD = []
            resD = [0]

            initRandom(&xArrD, sizeX, (0...10))
            initRandom(&yArrD, sizeY, (0...10))
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrD, M: yArrD.count, [.storageModeManaged])!
                resBuf = metalBlas.getDeviceBuffer(matA: resD, M: resD.count, [.storageModeManaged])
            }
        }
        else if T.self == Float16.self
        {
            xArrH = []
            yArrH = []
            resH = [0]

            initRandom(&xArrH, sizeX, (0...10))
            initRandom(&yArrH, sizeY, (0...10))
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrH, M: yArrH.count, [.storageModeManaged])!
                resBuf = metalBlas.getDeviceBuffer(matA: resH, M: resH.count, [.storageModeManaged])
            }
        }
    }

    private func callDot(_ callRef: Bool) -> T
    {
        if callRef
        {
            if T.self == Float.self
            {
                return T(refSdot(N, xArrF, incx, yArrF, incy))
            }
            else if T.self == Double.self
            {
                return T(refDdot(N, xArrD, incx, yArrD, incy))
            }
            else
            {
                return T(refHdot(N, xArrH, incx, yArrH, incy))
            }
        }
        else if useBuffers
        {
            if T.self == Float.self
            {
                metalBlas.metalSdot(N, xBuf, incx, yBuf, incy, &resBuf)
            }
            else if T.self == Double.self
            {
                metalBlas.metalDdot(N, xBuf, incx, yBuf, incy, &resBuf)
            }
            else
            {
                metalBlas.metalHdot(N, xBuf, incx, yBuf, incy, &resBuf)
            }
        }
        else
        {
            if T.self == Float.self
            {
                return T(metalBlas.metalSdot(N, xArrF, incx, yArrF, incy))
            }
            else if T.self == Double.self
            {
                return T(metalBlas.metalDdot(N, xArrD, incx, yArrD, incy))
            }
            else
            {
                return T(metalBlas.metalHdot(N, xArrH, incx, yArrH, incy))
            }
        }

        // Buffer dot will return -1, have to use buffer result
        return -1
    }

    func validateDot() -> Bool
    {
        if T.self == Float.self
        {
            let xCpy = xArrF!
            let yCpy = yArrF!
            let refRes = refSdot(N, xCpy, incx, yCpy, incy)
            let metRes = callDot(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(resBuf, &resF)
                return printIfNotNear(resF[0], refRes, N)
            }
            else
            {
                return printIfNotNear(Float(metRes), refRes, N)
            }
        }
        else if T.self == Double.self
        {
            let xCpy = xArrD!
            let yCpy = yArrD!
            let refRes = refDdot(N, xCpy, incx, yCpy, incy)
            let metRes = callDot(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(resBuf, &resD)
                return printIfNotNear(resD[0], refRes, N)
            }
            else
            {
                return printIfNotNear(Double(metRes), refRes, N)
            }
        }
        else if T.self == Float16.self
        {
            let xCpy = xArrH!
            let yCpy = yArrH!
            let refRes = refHdot(N, xCpy, incx, yCpy, incy)
            let metRes = callDot(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(resBuf, &resH)
                return printIfNotNear(resH[0], refRes, N)
            }
            else
            {
                return printIfNotNear(Float16(metRes), refRes, N)
            }
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkDot(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            _ = callDot(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    _ = callDot(benchRef)
                }
            }
        )
        return result
    }
}

class dotTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/dotInput"
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
        print("\tfunc,prec,N,incx,incy,useBuf,coldIters,hotIters\n\t", terminator: "")
        print(params.function, params.prec, params.N, params.incx, params.incy, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func dotTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let dotFramework = DotFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = dotFramework.validateDot()
            XCTAssert(pass)
        }

        var result = dotFramework.benchmarkDot(false, params.coldIters, params.hotIters)
        var accResult = dotFramework.benchmarkDot(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getDotGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getDotGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func dotLauncher()
    {
        if params.prec == .fp32
        {
            dotTester(Float(0))
        }
        else if params.prec == .fp64
        {
            dotTester(Double(0))
        }
        else if params.prec == .fp16
        {
            dotTester(Float16(0))
        }
    }

    func testDot0()
    {
        setupParams(0)
        printTestInfo("testDot0")
        dotLauncher()
    }

    func testDot1()
    {
        setupParams(1)
        printTestInfo("testDot1")
        dotLauncher()
    }

    func testDot2()
    {
        setupParams(2)
        printTestInfo("testDot2")
        dotLauncher()
    }

    func testDot3()
    {
        setupParams(3)
        printTestInfo("testDot3")
        dotLauncher()
    }
}

