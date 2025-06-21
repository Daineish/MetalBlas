//
//  tbsvTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-23.
//

import XCTest
@testable import MetalBlasFramework

class TbsvFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var aArrH : [ Float16 ]!
    var xArrH : [ Float16 ]!
    var solArrH : [ Float16 ]!
    var aArrF : [ Float ]!
    var xArrF : [ Float ]!
    var solArrF : [ Float ]!
    var aArrD : [ Double ]!
    var xArrD : [ Double ]!
    var solArrD : [ Double ]!
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
            initRandomMatrix(&aArrF, N, N, lda, order, (1...5), .DiagDominant)
            initRandomVec(&xArrF, N, incx, (1...5))

            // To simplify testing, calling tbmv so we have a perfect integer result to compare to
            solArrF = xArrF
            refStbmv(order, uplo, trans, diag, N, K, aArrF, lda, &xArrF, incx)

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrF, M: aArrF.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Double.self
        {
            aArrD = []; xArrD = []
            initRandomMatrix(&aArrD, N, N, lda, order, (1...5), .DiagDominant)
            initRandomVec(&xArrD, N, incx, (1...5))

            solArrD = xArrD
            refDtbmv(order, uplo, trans, diag, N, K, aArrD, lda, &xArrD, incx)

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrD, M: aArrD.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
        else if T.self == Float16.self
        {
            aArrH = []; xArrH = []
            initRandomMatrix(&aArrH, N, N, lda, order, (1...5), .DiagDominant)
            initRandomVec(&xArrH, N, incx, (1...5))

            solArrH = xArrH
            refHtbmv(order, uplo, trans, diag, N, K, aArrH, lda, &xArrH, incx)

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrH, M: aArrH.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
    }

    private func callTbsv(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refStbsv(order, uplo, trans, diag,  N, K, aArrF, lda, &xArrF, incx) :
            T.self == Double.self ? refDtbsv(order, uplo, trans, diag,  N, K, aArrD, lda, &xArrD, incx) :
                                    refHtbsv(order, uplo, trans, diag,  N, K, aArrH, lda, &xArrH, incx)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalStbsv(order, uplo, trans, diag,  N, K, aBuf, lda, &xBuf, incx) :
            T.self == Double.self ? metalBlas.metalDtbsv(order, uplo, trans, diag,  N, K, aBuf, lda, &xBuf, incx) :
                                    metalBlas.metalHtbsv(order, uplo, trans, diag,  N, K, aBuf, lda, &xBuf, incx)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalStbsv(order, uplo, trans, diag, N, K, aArrF, lda, &xArrF, incx) :
            T.self == Double.self ? metalBlas.metalDtbsv(order, uplo, trans, diag, N, K, aArrD, lda, &xArrD, incx) :
                                    metalBlas.metalHtbsv(order, uplo, trans, diag, N, K, aArrH, lda, &xArrH, incx)
        }
    }

    func validateTbsv() -> Bool
    {
        if T.self == Float.self
        {
            var xCpy = xArrF!
            refStbsv(order, uplo, trans, diag, N, K, aArrF, lda, &xCpy, incx)
            callTbsv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrF)
            }

            return printIfNotNear(xArrF, solArrF, N * N)
        }
        else if T.self == Double.self
        {
            var xCpy = xArrD!
            refDtbsv(order, uplo, trans, diag, N, K, aArrD, lda, &xCpy, incx)
            callTbsv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrD)
            }

            return printIfNotNear(xArrD, solArrD, N * N)
        }
        else if T.self == Float16.self
        {
            var xCpy = xArrH!
            refHtbsv(order, uplo, trans, diag, N, K, aArrH, lda, &xCpy, incx)
            callTbsv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrH)
            }

            return printIfNotNear(xArrH, solArrH, N * N)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkTbsv(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callTbsv(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callTbsv(benchRef)
                }
            }
        )
        return result
    }
}

class tbsvTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas2/tbsvInput"
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
        print("\ttbsv", params.prec, params.order, params.uplo,params.transA,params.diag,params.N, params.K, params.lda, params.incx,params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func tbsvTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let tbsvFramework = TbsvFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = tbsvFramework.validateTbsv()
            XCTAssert(pass)
        }

        var result = tbsvFramework.benchmarkTbsv(false, params.coldIters, params.hotIters)
        var accResult = tbsvFramework.benchmarkTbsv(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getTbsvGflopCount(N: params.N, K: params.K) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getTbsvGflopCount(N: params.N, K: params.K) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func tbsvLauncher()
    {
        if params.prec == precisionType.fp32
        {
            tbsvTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            tbsvTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            tbsvTester(Float16(0))
        }
    }

    func testTbsv0()
    {
        for i in 0..<105
        {
            setupParams(i)
            printTestInfo("testTbsv" + String(i))
            tbsvLauncher()
        }
    }
}

