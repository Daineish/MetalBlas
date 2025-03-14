//
//  tbmvTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

import XCTest
@testable import MetalBlasFramework

class TbmvFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var aArrH : [ Float16 ]!
    var xArrH : [ Float16 ]!
    var aArrF : [ Float ]!
    var xArrF : [ Float ]!
    var aArrD : [ Double ]!
    var xArrD : [ Double ]!
    let N : Int
    let K : Int
    let lda : Int
    let incx : Int

    let order : OrderType
    let uplo : UploType
    let trans : TransposeType
    let diag : DiagType

    var aBuf : MTLBuffer!
    var xBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        N = params.N
        K = params.K
        lda = params.lda
        incx = params.incx

        order = params.order
        uplo = params.uplo
        trans = params.transA
        diag = params.diag

        metalBlas = metalBlasIn!

        let sizeA : Int = N * lda
        let sizeX : Int = N * incx

        // Can initialize as if a normal matrix, will deal with it as if it were banded. This
        // way we'll implicitly check that we aren't changing 'out-of-band' data.
        if T.self == Float.self
        {
            aArrF = []; xArrF = []
            initRandomInt(&aArrF, sizeA)
            initRandomInt(&xArrF, sizeX)

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrF, M: aArrF.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Double.self
        {
            aArrD = []; xArrD = []
            initRandomInt(&aArrD, sizeA)
            initRandomInt(&xArrD, sizeX)

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrD, M: aArrD.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Float16.self
        {
            aArrH = []; xArrH = []
            initRandomInt(&aArrH, sizeA, (0...3))
            initRandomInt(&xArrH, sizeX, (0...3))
            
//            printMatrix(aArrH, N, N, lda, order)
//            print("\n")
//            printVector(xArrH, N, incx)

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrH, M: aArrH.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
    }

    private func callTbmv(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refStbmv(order, uplo, trans, diag,  N, K, aArrF, lda, &xArrF, incx) :
            T.self == Double.self ? refDtbmv(order, uplo, trans, diag,  N, K, aArrD, lda, &xArrD, incx) :
                                    refHtbmv(order, uplo, trans, diag,  N, K, aArrH, lda, &xArrH, incx)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalStbmv(order, uplo, trans, diag,  N, K, aBuf, lda, &xBuf, incx) :
            T.self == Double.self ? metalBlas.metalDtbmv(order, uplo, trans, diag,  N, K, aBuf, lda, &xBuf, incx) :
                                    metalBlas.metalHtbmv(order, uplo, trans, diag,  N, K, aBuf, lda, &xBuf, incx)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalStbmv(order, uplo, trans, diag, N, K, aArrF, lda, &xArrF, incx) :
            T.self == Double.self ? metalBlas.metalDtbmv(order, uplo, trans, diag, N, K, aArrD, lda, &xArrD, incx) :
                                    metalBlas.metalHtbmv(order, uplo, trans, diag, N, K, aArrH, lda, &xArrH, incx)
        }
    }

    func validateTbmv() -> Bool
    {
        if T.self == Float.self
        {
            var xCpy = xArrF!
            refStbmv(order, uplo, trans, diag, N, K, aArrF, lda, &xCpy, incx)
            callTbmv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrF)
            }

            return printIfNotEqual(xArrF, xCpy)
        }
        else if T.self == Double.self
        {
            var xCpy = xArrD!
            refDtbmv(order, uplo, trans, diag, N, K, aArrD, lda, &xCpy, incx)
            callTbmv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrD)
            }

            return printIfNotEqual(xArrD, xCpy)
        }
        else if T.self == Float16.self
        {
            var xCpy = xArrH!
            refHtbmv(order, uplo, trans, diag, N, K, aArrH, lda, &xCpy, incx)
            callTbmv(false)
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

    func benchmarkTbmv(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callTbmv(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callTbmv(benchRef)
                }
            }
        )
        return result
    }
}

class tbmvTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas2/tbmvInput"
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
        print("\tfunc,prec,order,uplo,trans,diag,N,K,lda,incx,useBuf,coldIters,hotIters")
        print("\ttbmv", params.prec, params.order, params.uplo,params.transA,params.diag,params.N, params.K, params.lda, params.incx,params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func tbmvTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let tbmvFramework = TbmvFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = tbmvFramework.validateTbmv()
            XCTAssert(pass)
        }

        var result = tbmvFramework.benchmarkTbmv(false, params.coldIters, params.hotIters)
        var accResult = tbmvFramework.benchmarkTbmv(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getTbmvGflopCount(N: params.N, K: params.K) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getTbmvGflopCount(N: params.N, K: params.K) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func tbmvLauncher()
    {
        if params.prec == precisionType.fp32
        {
            tbmvTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            tbmvTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            tbmvTester(Float16(0))
        }
    }

    func testTbmv0()
    {
        for i in 0..<102
        {
            setupParams(i)
            printTestInfo("testTbmv" + String(i))
            tbmvLauncher()
        }
    }
}

