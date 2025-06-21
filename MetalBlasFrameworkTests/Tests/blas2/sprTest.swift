//
//  sprTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-10.
//

import XCTest
@testable import MetalBlasFramework

class SprFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var aArrH : [ Float16 ]!
    var xArrH : [ Float16 ]!
    var aArrF : [ Float ]!
    var xArrF : [ Float ]!
    var aArrD : [ Double ]!
    var xArrD : [ Double ]!
    var alpha: Float
    let N : Int
    let incx : Int

    let order : OrderType
    let uplo : UploType

    var aBuf : MTLBuffer!
    var xBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        alpha = Float(params.alpha)
        N = params.N
        incx = params.incx

        order = params.order
        uplo = params.uplo

        metalBlas = metalBlasIn!

        let sizeA : Int = N * (N + 1) / 2
        let sizeX : Int = N * incx

        alpha = params.alpha

        // Can initialize as if a normal matrix, will deal with it as if it were banded. This
        // way we'll implicitly check that we aren't changing 'out-of-band' data.
        if T.self == Float.self
        {
            aArrF = []; xArrF = []
            initRandomVec(&aArrF, sizeA, 1)
            initRandomVec(&xArrF, N, incx)

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrF, M: aArrF.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Double.self
        {
            aArrD = []; xArrD = []
            initRandomVec(&aArrD, sizeA, 1)
            initRandomVec(&xArrD, N, incx)
            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrD, M: aArrD.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Float16.self
        {
            aArrH = []; xArrH = []
            initRandomVec(&aArrH, sizeA, 1, (0...3))
            initRandomVec(&xArrH, N, incx, (0...3))

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrH, M: aArrH.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
    }

    private func callSpr(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSspr(order, uplo, N, alpha, xArrF, incx, &aArrF) :
            T.self == Double.self ? refDspr(order, uplo, N, Double(alpha), xArrD, incx, &aArrD) :
                                    refHspr(order, uplo, N, Float16(alpha), xArrH, incx, &aArrH)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSspr(order, uplo, N, alpha, xBuf, incx, &aBuf) :
            T.self == Double.self ? metalBlas.metalDspr(order, uplo, N, Double(alpha), xBuf, incx, &aBuf) :
                                    metalBlas.metalHspr(order, uplo, N, Float16(alpha), xBuf, incx, &aBuf)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSspr(order, uplo, N, alpha, xArrF, incx, &aArrF) :
            T.self == Double.self ? metalBlas.metalDspr(order, uplo, N, Double(alpha), xArrD, incx, &aArrD) :
                                    metalBlas.metalHspr(order, uplo,N, Float16(alpha), xArrH, incx, &aArrH)
        }
    }

    func validateSpr() -> Bool
    {
        if T.self == Float.self
        {
            var aCpy = aArrF!
            refSspr(order, uplo, N, alpha, xArrF, incx, &aCpy)
            callSpr(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrF)
            }

            return printIfNotEqual(aArrF, aCpy)
        }
        else if T.self == Double.self
        {
            var aCpy = aArrD!
            refDspr(order, uplo, N, Double(alpha), xArrD, incx, &aCpy)
            callSpr(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrD)
            }

            return printIfNotEqual(aArrD, aCpy)
        }
        else if T.self == Float16.self
        {
            var aCpy = aArrH!
            refHspr(order, uplo, N, Float16(alpha), xArrH, incx, &aCpy)
            callSpr(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrH)
            }

            return printIfNotEqual(aArrH, aCpy)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkSpr(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callSpr(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callSpr(benchRef)
                }
            }
        )
        return result
    }
}

class sprTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas2/sprInput"
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
        print("\tfunc,prec,order,uplo,N,incx,alpha,useBuf,coldIters,hotIters")
        print("\tspr", params.prec, params.order, params.uplo, params.N, params.incx, params.alpha, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func sprTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let sprFramework = SprFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = sprFramework.validateSpr()
            XCTAssert(pass)
        }

        var result = sprFramework.benchmarkSpr(false, params.coldIters, params.hotIters)
        var accResult = sprFramework.benchmarkSpr(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getSprGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getSprGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func sprLauncher()
    {
        if params.prec == precisionType.fp32
        {
            sprTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            sprTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            sprTester(Float16(0))
        }
    }

    func testSpr0()
    {
        for i in 0..<64
        {
            setupParams(i)
            printTestInfo("testSpr" + String(i))
            sprLauncher()
        }
    }
}

