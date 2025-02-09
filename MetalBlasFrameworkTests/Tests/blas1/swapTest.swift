//
//  swapTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

import XCTest
@testable import MetalBlasFramework

class SwapFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var yArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var yArrF : [ Float ]!
    var xArrD : [ Double ]!
    var yArrD : [ Double ]!
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

        metalBlas = metalBlasIn!

        let sizeX : Int = params.N * params.incx
        let sizeY : Int = params.N * params.incy

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

    private func callSwap(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSswap(N, &xArrF, incx, &yArrF, incy) :
            T.self == Double.self ? refDswap(N, &xArrD, incx, &yArrD, incy) :
                                    refHswap(N, &xArrH, incx, &yArrH, incy)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSswap(N, &xBuf, incx, &yBuf, incy) :
            T.self == Double.self ? metalBlas.metalDswap(N, &xBuf, incx, &yBuf, incy) :
                                    metalBlas.metalHswap(N, &xBuf, incx, &yBuf, incy)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSswap(N, &xArrF, incx, &yArrF, incy) :
            T.self == Double.self ? metalBlas.metalDswap(N, &xArrD, incx, &yArrD, incy) :
                                    metalBlas.metalHswap(N, &xArrH, incx, &yArrH, incy)
        }
    }

    func validateSwap() -> Bool
    {
        if T.self == Float.self
        {
            var xCpy = xArrF!
            var yCpy = yArrF!
            refSswap(N, &xCpy, incx, &yCpy, incy)
            callSwap(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrF)
                metalBlas.copyBufToArray(yBuf, &yArrF)
            }

            return (printIfNotEqual(yArrF, yCpy) && printIfNotEqual(xArrF, xCpy))
        }
        else if T.self == Double.self
        {
            var xCpy = xArrD!
            var yCpy = yArrD!
            refDswap(N, &xCpy, incx, &yCpy, incy)
            callSwap(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrD)
                metalBlas.copyBufToArray(yBuf, &yArrD)
            }

            return (printIfNotEqual(yArrD, yCpy) && printIfNotEqual(xArrD, xCpy))
        }
        else if T.self == Float16.self
        {
            var xCpy = xArrH!
            var yCpy = yArrH!
            refHswap(N, &xCpy, incx, &yCpy, incy)
            callSwap(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrH)
                metalBlas.copyBufToArray(yBuf, &yArrH)
            }

            
            return (printIfNotEqual(yArrH, yCpy) && printIfNotEqual(xArrH, xCpy))
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkSwap(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callSwap(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callSwap(benchRef)
                }
            }
        )
        return result
    }
}

class swapTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas1/swapInput"
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
        print("\tfunc,prec,N,incx,incy,useBuf,coldIters,hotIters")
        print("\tswap", params.prec, params.N, params.incx, params.incy, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func swapTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let swapFramework = SwapFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = swapFramework.validateSwap()
            XCTAssert(pass)
        }

        var result = swapFramework.benchmarkSwap(false, params.coldIters, params.hotIters)
        var accResult = swapFramework.benchmarkSwap(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gbytes = getSwapGbyteCount(N: params.N) * Double(MemoryLayout<T>.stride) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gbytes: ", gbytes)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgbytes = getSwapGbyteCount(N: params.N) * Double(MemoryLayout<T>.stride) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gbytes: ", accgbytes)
        }
    }

    func swapLaunhcer()
    {
        if params.prec == precisionType.fp32
        {
            swapTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            swapTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            swapTester(Float16(0))
        }
    }

    func testSwap0()
    {
        // TODO: test more than one line per test? Probably on a per-file basis or something
        setupParams(0)
        printTestInfo("testSwap0")
        swapLaunhcer()
    }

    func testSwap1()
    {
        setupParams(1)
        printTestInfo("testSwap1")
        swapLaunhcer()
    }

    func testSwap2()
    {
        setupParams(2)
        printTestInfo("testSwap2")
        swapLaunhcer()
    }

    func testSwap3()
    {
        setupParams(3)
        printTestInfo("testSwap3")
        swapLaunhcer()
    }
}

