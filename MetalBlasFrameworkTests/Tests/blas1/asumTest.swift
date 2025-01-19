//
//  asumTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-17.
//

import XCTest
@testable import MetalBlasFramework

class AsumFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var xArrD : [ Double ]!
    var resH : [ Float16 ]!
    var resF : [ Float ]!
    var resD : [ Double ]!
    
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

        if T.self == Float.self
        {
            xArrF = []
            resF = [0]
            
            initRandom(&xArrF, sizeX, (0...0.1))

            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
                resBuf = metalBlas.getDeviceBuffer(matA: resF, M: resF.count, [.storageModeManaged])
            }
        }
        else if T.self == Double.self
        {
            xArrD = []
            resD = [0]

            initRandom(&xArrD, sizeX, (0...0.1))
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
                resBuf = metalBlas.getDeviceBuffer(matA: resD, M: resD.count, [.storageModeManaged])
            }
        }
        else if T.self == Float16.self
        {
            xArrH = []
            resH = [0]

            initRandom(&xArrH, sizeX, (0...0.1))
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
                resBuf = metalBlas.getDeviceBuffer(matA: resH, M: resH.count, [.storageModeManaged])
            }
        }
    }

    private func callAsum(_ callRef: Bool) -> T
    {
        if callRef
        {
            if T.self == Float.self
            {
                return T(refSasum(N, xArrF, incx))
            }
            else if T.self == Double.self
            {
                return T(refDasum(N, xArrD, incx))
            }
            else
            {
                return T(refHasum(N, xArrH, incx))
            }
        }
        else if useBuffers
        {
            if T.self == Float.self
            {
                metalBlas.metalSasum(N, xBuf, incx, &resBuf)
            }
            else if T.self == Double.self
            {
                metalBlas.metalDasum(N, xBuf, incx, &resBuf)
            }
            else
            {
                metalBlas.metalHasum(N, xBuf, incx, &resBuf)
            }
        }
        else
        {
            if T.self == Float.self
            {
                return T(metalBlas.metalSasum(N, xArrF, incx))
            }
            else if T.self == Double.self
            {
                return T(metalBlas.metalDasum(N, xArrD, incx))
            }
            else
            {
                return T(metalBlas.metalHasum(N, xArrH, incx))
            }
        }

        // Buffer asum will return -1, have to use buffer result
        return -1
    }

    func validateAsum() -> Bool
    {
        if T.self == Float.self
        {
            let xCpy = xArrF!
            let refRes = refSasum(N, xCpy, incx)
            let metRes = callAsum(false)
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
            let refRes = refDasum(N, xCpy, incx)
            let metRes = callAsum(false)
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
            let refRes = refHasum(N, xCpy, incx)
            let metRes = callAsum(false)
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

    func benchmarkAsum(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            _ = callAsum(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    _ = callAsum(benchRef)
                }
            }
        )
        return result
    }
}

class asumTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/asumInput"
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

    func asumTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let asumFramework = AsumFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = asumFramework.validateAsum()
            XCTAssert(pass)
        }

        var result = asumFramework.benchmarkAsum(false, params.coldIters, params.hotIters)
        var accResult = asumFramework.benchmarkAsum(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getAsumGflopCount(N: params.N) / avgTime
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

    func asumLauncher()
    {
        if params.prec == .fp32
        {
            asumTester(Float(0))
        }
        else if params.prec == .fp64
        {
            asumTester(Double(0))
        }
        else if params.prec == .fp16
        {
            asumTester(Float16(0))
        }
    }

    func testAsum0()
    {
        setupParams(0)
        printTestInfo("testAsum0")
        asumLauncher()
    }

    func testAsum1()
    {
        setupParams(1)
        printTestInfo("testAsum1")
        asumLauncher()
    }

    func testAsum2()
    {
        setupParams(2)
        printTestInfo("testAsum2")
        asumLauncher()
    }

    func testAsum3()
    {
        setupParams(3)
        printTestInfo("testAsum3")
        asumLauncher()
    }
}

