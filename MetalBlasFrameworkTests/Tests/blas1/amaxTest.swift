//
//  amaxTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-20.
//

import XCTest
@testable import MetalBlasFramework

class AmaxFramework<T: Numeric>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var xArrD : [ Double ]!
    var resM : [ Int ]!
    
    let N : Int
    let incx : Int

    var xBuf : MTLBuffer!
    var resBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        N = params.N
        incx = params.incx

        metalBlas = metalBlasIn!

        let sizeX : Int = params.N * params.incx
        resM = [0]
        if useBuffers
        {
            resBuf = metalBlas.getDeviceBuffer(matA: resM, M: resM.count, [.storageModeManaged])
        }

        if T.self == Float.self
        {
            xArrF = []
            initRandomVec(&xArrF, N, incx, (-64000...64000))
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Double.self
        {
            xArrD = []
            initRandomVec(&xArrD, N, incx, (-64000...64000))
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Float16.self
        {
            xArrH = []
            initRandomVec(&xArrH, N, incx, (-20000...20000))
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
    }

    private func callAmax(_ callRef: Bool) -> Int
    {
        if callRef
        {
            if T.self == Float.self
            {
                return refIsamax(N, xArrF, incx)
            }
            else if T.self == Double.self
            {
                return refIdamax(N, xArrD, incx)
            }
            else
            {
                return refIhamax(N, xArrH, incx)
            }
        }
        else if useBuffers
        {
            if T.self == Float.self
            {
                metalBlas.metalIsamax(N, xBuf, incx, &resBuf)
            }
            else if T.self == Double.self
            {
                metalBlas.metalIdamax(N, xBuf, incx, &resBuf)
            }
            else
            {
                metalBlas.metalIhamax(N, xBuf, incx, &resBuf)
            }
        }
        else
        {
            if T.self == Float.self
            {
                return metalBlas.metalIsamax(N, xArrF, incx)
            }
            else if T.self == Double.self
            {
                return metalBlas.metalIdamax(N, xArrD, incx)
            }
            else
            {
                return metalBlas.metalIhamax(N, xArrH, incx)
            }
        }

        // Buffer amax will return -1, have to use buffer result
        return -1
    }

    func validateAmax() -> Bool
    {
        var refRes : Int
        if T.self == Float.self
        {
            let xCpy = xArrF!
            refRes = refIsamax(N, xCpy, incx)
        }
        else if T.self == Double.self
        {
            let xCpy = xArrD!
            refRes = refIdamax(N, xCpy, incx)
        }
        else if T.self == Float16.self
        {
            let xCpy = xArrH!
            refRes = refIhamax(N, xCpy, incx)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }

        if useBuffers
        {
            _ = callAmax(false)
            metalBlas.copyBufToArray(resBuf, &resM)
            return printIfNotEqual(resM[0], refRes)
        }
        else
        {
            let metRes = callAmax(false)
            return printIfNotEqual(metRes, refRes)
        }
    }

    func benchmarkAmax(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            _ = callAmax(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    _ = callAmax(benchRef)
                }
            }
        )
        return result
    }
}

class amaxTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas1/amaxInput"
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
        print("\tfunc,prec,N,incx,useBuf,coldIters,hotIters\n\t", terminator: "")
        print(params.function, params.prec, params.N, params.incx, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func amaxTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let amaxFramework = AmaxFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = amaxFramework.validateAmax()
            XCTAssert(pass)
        }

        var result = amaxFramework.benchmarkAmax(false, params.coldIters, params.hotIters)
        var accResult = amaxFramework.benchmarkAmax(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getAmaxGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getAmaxGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func amaxLauncher()
    {
        if params.prec == .fp32
        {
            amaxTester(Float(0))
        }
        else if params.prec == .fp64
        {
            amaxTester(Double(0))
        }
        else if params.prec == .fp16
        {
            amaxTester(Float16(0))
        }
    }

    func testAmax0()
    {
        setupParams(0)
        printTestInfo("testAmax0")
        amaxLauncher()
    }

    func testAmax1()
    {
        setupParams(1)
        printTestInfo("testAmax1")
        amaxLauncher()
    }

    func testAmax2()
    {
        setupParams(2)
        printTestInfo("testAmax2")
        amaxLauncher()
    }

    func testAmax3()
    {
        setupParams(3)
        printTestInfo("testAmax3")
        amaxLauncher()
    }
}

