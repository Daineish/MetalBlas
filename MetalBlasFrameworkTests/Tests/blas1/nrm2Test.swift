//
//  nrm2Test.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-19.
//

import XCTest
@testable import MetalBlasFramework

class Nrm2Framework<T: BinaryFloatingPoint>
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

    private func callNrm2(_ callRef: Bool) -> T
    {
        if callRef
        {
            if T.self == Float.self
            {
                return T(refSnrm2(N, xArrF, incx))
            }
            else if T.self == Double.self
            {
                return T(refDnrm2(N, xArrD, incx))
            }
            else
            {
                return T(refHnrm2(N, xArrH, incx))
            }
        }
        else if useBuffers
        {
            if T.self == Float.self
            {
                metalBlas.metalSnrm2(N, xBuf, incx, &resBuf)
            }
            else if T.self == Double.self
            {
                metalBlas.metalDnrm2(N, xBuf, incx, &resBuf)
            }
            else
            {
                metalBlas.metalHnrm2(N, xBuf, incx, &resBuf)
            }
        }
        else
        {
            if T.self == Float.self
            {
                return T(metalBlas.metalSnrm2(N, xArrF, incx))
            }
            else if T.self == Double.self
            {
                return T(metalBlas.metalDnrm2(N, xArrD, incx))
            }
            else
            {
                return T(metalBlas.metalHnrm2(N, xArrH, incx))
            }
        }

        // Buffer nrm2 will return -1, have to use buffer result
        return -1
    }

    func validateNrm2() -> Bool
    {
        if T.self == Float.self
        {
            let xCpy = xArrF!
            let refRes = refSnrm2(N, xCpy, incx)
            let metRes = callNrm2(false)
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
            let refRes = refDnrm2(N, xCpy, incx)
            let metRes = callNrm2(false)
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
            let refRes = refHnrm2(N, xCpy, incx)
            let metRes = callNrm2(false)
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

    func benchmarkNrm2(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            _ = callNrm2(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    _ = callNrm2(benchRef)
                }
            }
        )
        return result
    }
}

class nrm2Test: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/nrm2Input"
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

    func nrm2Tester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let nrm2Framework = Nrm2Framework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = nrm2Framework.validateNrm2()
            XCTAssert(pass)
        }

        var result = nrm2Framework.benchmarkNrm2(false, params.coldIters, params.hotIters)
        var accResult = nrm2Framework.benchmarkNrm2(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getNrm2GflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getNrm2GflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func nrm2Launcher()
    {
        if params.prec == .fp32
        {
            nrm2Tester(Float(0))
        }
        else if params.prec == .fp64
        {
            nrm2Tester(Double(0))
        }
        else if params.prec == .fp16
        {
            nrm2Tester(Float16(0))
        }
    }

    func testNrm20()
    {
        setupParams(0)
        printTestInfo("testNrm20")
        nrm2Launcher()
    }

    func testNrm21()
    {
        setupParams(1)
        printTestInfo("testNrm21")
        nrm2Launcher()
    }

    func testNrm22()
    {
        setupParams(2)
        printTestInfo("testNrm22")
        nrm2Launcher()
    }

    func testNrm23()
    {
        setupParams(3)
        printTestInfo("testNrm23")
        nrm2Launcher()
    }
}

