//
//  syrTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

import XCTest
@testable import MetalBlasFramework

class SyrFramework<T: BinaryFloatingPoint>
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
    let lda : Int

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
        lda = params.lda

        order = params.order
        uplo = params.uplo

        metalBlas = metalBlasIn!

        let sizeA : Int = N * lda
        let sizeX : Int = N * incx

        alpha = params.alpha

        // Can initialize as if a normal matrix, will deal with it as if it were banded. This
        // way we'll implicitly check that we aren't changing 'out-of-band' data.
        if T.self == Float.self
        {
            aArrF = []; xArrF = []
            initRandomMatrix(&aArrF, N, N, lda, order)
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
            initRandomMatrix(&aArrD, N, N, lda, order)
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
            initRandomMatrix(&aArrH, N, N, lda, order, (0...3))
            initRandomVec(&xArrH, N, incx, (0...3))

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrH, M: aArrH.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
    }

    private func callSyr(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSsyr(order, uplo, N, alpha, xArrF, incx, &aArrF, lda) :
            T.self == Double.self ? refDsyr(order, uplo, N, Double(alpha), xArrD, incx, &aArrD, lda) :
                                    refHsyr(order, uplo, N, Float16(alpha), xArrH, incx, &aArrH, lda)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSsyr(order, uplo, N, alpha, xBuf, incx, &aBuf, lda) :
            T.self == Double.self ? metalBlas.metalDsyr(order, uplo, N, Double(alpha), xBuf, incx, &aBuf, lda) :
                                    metalBlas.metalHsyr(order, uplo, N, Float16(alpha), xBuf, incx, &aBuf, lda)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSsyr(order, uplo, N, alpha, xArrF, incx, &aArrF, lda) :
            T.self == Double.self ? metalBlas.metalDsyr(order, uplo, N, Double(alpha), xArrD, incx, &aArrD, lda) :
                                    metalBlas.metalHsyr(order, uplo,N, Float16(alpha), xArrH, incx, &aArrH, lda)
        }
    }

    func validateSyr() -> Bool
    {
        if T.self == Float.self
        {
            var aCpy = aArrF!
            refSsyr(order, uplo, N, alpha, xArrF, incx, &aCpy, lda)
            callSyr(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrF)
            }

            return printIfNotEqual(aArrF, aCpy)
        }
        else if T.self == Double.self
        {
            var aCpy = aArrD!
            refDsyr(order, uplo, N, Double(alpha), xArrD, incx, &aCpy, lda)
            callSyr(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrD)
            }

            return printIfNotEqual(aArrD, aCpy)
        }
        else if T.self == Float16.self
        {
            var aCpy = aArrH!
            refHsyr(order, uplo, N, Float16(alpha), xArrH, incx, &aCpy, lda)
            callSyr(false)
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

    func benchmarkSyr(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callSyr(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callSyr(benchRef)
                }
            }
        )
        return result
    }
}

class syrTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas2/syrInput"
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
        print("\tfunc,prec,order,uplo,N,incx,lda,alpha,useBuf,coldIters,hotIters")
        print("\tsyr", params.prec, params.order, params.uplo, params.N, params.incx, params.lda, params.alpha, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func syrTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let syrFramework = SyrFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = syrFramework.validateSyr()
            XCTAssert(pass)
        }

        var result = syrFramework.benchmarkSyr(false, params.coldIters, params.hotIters)
        var accResult = syrFramework.benchmarkSyr(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getSyrGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getSyrGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func syrLauncher()
    {
        if params.prec == precisionType.fp32
        {
            syrTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            syrTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            syrTester(Float16(0))
        }
    }

    func testSyr0()
    {
        for i in 0..<64
        {
            setupParams(i)
            printTestInfo("testSyr" + String(i))
            syrLauncher()
        }
    }
}

