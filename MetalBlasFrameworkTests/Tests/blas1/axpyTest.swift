//
//  axpyTests.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import XCTest
@testable import MetalBlasFramework

class AxpyFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var yArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var yArrF : [ Float ]!
    var xArrD : [ Double ]!
    var yArrD : [ Double ]!
    var alpha: Float
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
        alpha = Float(params.alpha)
        N = params.N
        incx = params.incx
        incy = params.incy

        metalBlas = metalBlasIn!

        let sizeX : Int = params.N * params.incx
        let sizeY : Int = params.N * params.incy

        alpha = params.alpha

        if T.self == Float.self
        {
            xArrF = []; yArrF = []
            initRandom(&xArrF, sizeX)
            initRandom(&yArrF, sizeY)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrF, M: yArrF.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
        else if T.self == Double.self
        {
            xArrD = []; yArrD = []
            initRandom(&xArrD, sizeX)
            initRandom(&yArrD, sizeY)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrD, M: yArrD.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
        else if T.self == Float16.self
        {
            xArrH = []; yArrH = []
            initRandom(&xArrH, sizeX)
            initRandom(&yArrH, sizeY)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrH, M: yArrH.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
    }

    private func callAxpy(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSaxpy(N, alpha, xArrF, incx, &yArrF, incy) :
            T.self == Double.self ? refDaxpy(N, Double(alpha), xArrD, incx, &yArrD, incy) :
                                    refHaxpy(N, Float16(alpha), xArrH, incx, &yArrH, incy)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSaxpy(N, alpha, xBuf, incx, &yBuf, incy) :
            T.self == Double.self ? metalBlas.metalDaxpy(N, Double(alpha), xBuf, incx, &yBuf, incy) :
                                    metalBlas.metalHaxpy(N, Float16(alpha), xBuf, incx, &yBuf, incy)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSaxpy(N, alpha, xArrF, incx, &yArrF, incy) :
            T.self == Double.self ? metalBlas.metalDaxpy(N, Double(alpha), xArrD, incx, &yArrD, incy) :
                                    metalBlas.metalHaxpy(N, Float16(alpha), xArrH, incx, &yArrH, incy)
        }
    }

    func validateAxpy() -> Bool
    {
        if T.self == Float.self
        {
            var yCpy = yArrF!
            refSaxpy(N, alpha, xArrF, incx, &yCpy, incy)
            callAxpy(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrF)
            }

            return printIfNotEqual(yArrF, yCpy)
        }
        else if T.self == Double.self
        {
            var yCpy = yArrD!
            refDaxpy(N, Double(alpha), xArrD, incx, &yCpy, incy)
            callAxpy(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrD)
            }

            return printIfNotEqual(yArrD, yCpy)
        }
        else if T.self == Float16.self
        {
            var yCpy = yArrH!
            refHaxpy(N, Float16(alpha), xArrH, incx, &yCpy, incy)
            callAxpy(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrH)
            }

            return printIfNotEqual(yArrH, yCpy)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkAxpy(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callAxpy(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callAxpy(benchRef)
                }
            }
        )
        return result
    }
}

class axpyTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas1/axpyInput"
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
        print("\tfunc,prec,N,incx,incy,alpha,useBuf,coldIters,hotIters")
        print("\taxpy", params.prec, params.N, params.incx, params.incy, params.alpha, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func axpyTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let axpyFramework = AxpyFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = axpyFramework.validateAxpy()
            XCTAssert(pass)
        }

        var result = axpyFramework.benchmarkAxpy(false, params.coldIters, params.hotIters)
        var accResult = axpyFramework.benchmarkAxpy(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getAxpyGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getAxpyGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func axpyLauncher()
    {
        if params.prec == precisionType.fp32
        {
            axpyTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            axpyTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            axpyTester(Float16(0))
        }
    }

    func testAxpy0()
    {
        // TODO: test more than one line per test? Probably on a per-file basis or something
        setupParams(0)
        printTestInfo("testAxpy0")
        axpyLauncher()
    }

    func testAxpy1()
    {
        setupParams(1)
        printTestInfo("testAxpy1")
        axpyLauncher()
    }

    func testAxpy2()
    {
        setupParams(2)
        printTestInfo("testAxpy2")
        axpyLauncher()
    }

    func testAxpy3()
    {
        setupParams(3)
        printTestInfo("testAxpy3")
        axpyLauncher()
    }
}
