//
//  scalTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import XCTest
@testable import MetalBlasFramework

class ScalFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var xArrD : [ Double ]!
    var alpha: Float
    let N : Int
    let incx : Int

    var xBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        alpha = Float(params.alpha)
        N = params.N
        incx = params.incx

        metalBlas = metalBlasIn!

        let sizeX : Int = params.N * params.incx

        alpha = params.alpha

        if T.self == Float.self
        {
            xArrF = []
            initRandomVec(&xArrF, N, incx)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Double.self
        {
            xArrD = []
            initRandomVec(&xArrD, N, incx)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Float16.self
        {
            xArrH = []
            initRandomVec(&xArrH, N, incx)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
    }

    private func callScal(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSscal(N, alpha, &xArrF, incx) :
            T.self == Double.self ? refDscal(N, Double(alpha), &xArrD, incx) :
                                    refHscal(N, Float16(alpha), &xArrH, incx)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSscal(N, alpha, &xBuf, incx) :
            T.self == Double.self ? metalBlas.metalDscal(N, Double(alpha), &xBuf, incx) :
                                    metalBlas.metalHscal(N, Float16(alpha), &xBuf, incx)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSscal(N, alpha, &xArrF, incx) :
            T.self == Double.self ? metalBlas.metalDscal(N, Double(alpha), &xArrD, incx) :
                                    metalBlas.metalHscal(N, Float16(alpha), &xArrH, incx)
        }
    }

    func validateScal() -> Bool
    {
        if T.self == Float.self
        {
            var xCpy = xArrF!
            refSscal(N, alpha, &xCpy, incx)
            callScal(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrF)
            }

            return printIfNotEqual(xArrF, xCpy)
        }
        else if T.self == Double.self
        {
            var xCpy = xArrD!
            refDscal(N, Double(alpha), &xCpy, incx)
            callScal(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrD)
            }

            return printIfNotEqual(xArrD, xCpy)
        }
        else if T.self == Float16.self
        {
            var xCpy = xArrH!
            refHscal(N, Float16(alpha), &xCpy, incx)
            callScal(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrH)
            }

            return printIfNotEqual(xArrH, xCpy)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkScal(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callScal(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callScal(benchRef)
                }
            }
        )
        return result
    }
}

class scalTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas1/scalInput"
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
        print("\tfunc,prec,N,incx,alpha,useBuf,coldIters,hotIters")
        print("\tscal", params.prec, params.N, params.incx, params.alpha, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func scalTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let scalFramework = ScalFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = scalFramework.validateScal()
            XCTAssert(pass)
        }

        var result = scalFramework.benchmarkScal(false, params.coldIters, params.hotIters)
        var accResult = scalFramework.benchmarkScal(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getScalGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getScalGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func scalLauncher()
    {
        if params.prec == .fp32
        {
            scalTester(Float(0))
        }
        else if params.prec == .fp64
        {
            scalTester(Double(0))
        }
        else if params.prec == .fp16
        {
            scalTester(Float16(0))
        }
    }

    func testScal0()
    {
        setupParams(0)
        printTestInfo("testScal0")
        scalLauncher()
    }

    func testScal1()
    {
        setupParams(1)
        printTestInfo("testScal1")
        scalLauncher()
    }

    func testScal2()
    {
        setupParams(2)
        printTestInfo("testScal2")
        scalLauncher()
    }

    func testScal3()
    {
        setupParams(3)
        printTestInfo("testScal3")
        scalLauncher()
    }
}
