//
//  rotTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-26.
//

import XCTest
@testable import MetalBlasFramework

class RotFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var yArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var yArrF : [ Float ]!
    var xArrD : [ Double ]!
    var yArrD : [ Double ]!
    var c : Float
    var s : Float
    let N : Int
    let incx : Int
    let incy : Int

    var xBuf : MTLBuffer!
    var yBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        N = params.N
        incx = params.incx
        incy = params.incy
        c = params.alpha
        s = params.beta

        metalBlas = metalBlasIn!

        let sizeX : Int = params.N * params.incx
        let sizeY : Int = params.N * params.incy

        if T.self == Float.self
        {
            xArrF = []; yArrF = []
            initRandomVec(&xArrF, N, incx)
            initRandomVec(&yArrF, N, incy)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrF, M: yArrF.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
        else if T.self == Double.self
        {
            xArrD = []; yArrD = []
            initRandomVec(&xArrD, N, incx)
            initRandomVec(&yArrD, N, incy)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrD, M: yArrD.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
        else if T.self == Float16.self
        {
            xArrH = []; yArrH = []
            initRandomVec(&xArrH, N, incx)
            initRandomVec(&yArrH, N, incy)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrH, M: yArrH.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
    }

    private func callRot(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSrot(N, &xArrF, incx, &yArrF, incy, c, s) :
            T.self == Double.self ? refDrot(N, &xArrD, incx, &yArrD, incy, Double(c), Double(s)) :
                                    refHrot(N, &xArrH, incx, &yArrH, incy, Float16(c), Float16(s))
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSrot(N, &xBuf, incx, &yBuf, incy, c, s) :
            T.self == Double.self ? metalBlas.metalDrot(N, &xBuf, incx, &yBuf, incy, Double(c), Double(s)) :
                                    metalBlas.metalHrot(N, &xBuf, incx, &yBuf, incy, Float16(c), Float16(s))
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSrot(N, &xArrF, incx, &yArrF, incy, c, s) :
            T.self == Double.self ? metalBlas.metalDrot(N, &xArrD, incx, &yArrD, incy, Double(c), Double(s)) :
                                    metalBlas.metalHrot(N, &xArrH, incx, &yArrH, incy, Float16(c), Float16(s))
        }
    }

    func validateRot() -> Bool
    {
        if T.self == Float.self
        {
            var xCpy = xArrF!
            var yCpy = yArrF!
            refSrot(N, &xCpy, incx, &yCpy, incy, c, s)
            callRot(false)
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
            refDrot(N, &xCpy, incx, &yCpy, incy, Double(c), Double(s))
            callRot(false)
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
            refHrot(N, &xCpy, incx, &yCpy, incy, Float16(c), Float16(s))
            callRot(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrH)
                metalBlas.copyBufToArray(yBuf, &yArrH)
            }

            return printIfNotEqual(yArrH, yCpy) && printIfNotEqual(xArrH, xCpy)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkRot(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callRot(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callRot(benchRef)
                }
            }
        )
        return result
    }
}

class rotTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas1/rotInput"
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
        print("\tfunc,prec,N,incx,incy,c,s,useBuf,coldIters,hotIters")
        print("\trot", params.prec, params.N, params.incx, params.incy, params.alpha, params.beta,params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func rotTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let rotFramework = RotFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = rotFramework.validateRot()
            XCTAssert(pass)
        }

        var result = rotFramework.benchmarkRot(false, params.coldIters, params.hotIters)
        var accResult = rotFramework.benchmarkRot(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getRotGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getRotGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func rotLauncher()
    {
        if params.prec == precisionType.fp32
        {
            rotTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            rotTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            rotTester(Float16(0))
        }
    }

    func testRot0()
    {
        // TODO: test more than one line per test? Probably on a per-file basis or something
        setupParams(0)
        printTestInfo("testRot0")
        rotLauncher()
    }

    func testRot1()
    {
        setupParams(1)
        printTestInfo("testRot1")
        rotLauncher()
    }

    func testRot2()
    {
        setupParams(2)
        printTestInfo("testRot2")
        rotLauncher()
    }

    func testRot3()
    {
        setupParams(3)
        printTestInfo("testRot3")
        rotLauncher()
    }
}

