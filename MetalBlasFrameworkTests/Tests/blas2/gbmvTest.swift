//
//  gbmvTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-03.
//

import XCTest
@testable import MetalBlasFramework

class GbmvFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var aArrH : [ Float16 ]!
    var xArrH : [ Float16 ]!
    var yArrH : [ Float16 ]!
    var aArrF : [ Float ]!
    var xArrF : [ Float ]!
    var yArrF : [ Float ]!
    var aArrD : [ Double ]!
    var xArrD : [ Double ]!
    var yArrD : [ Double ]!
    var alpha: Float
    var beta: Float
    let M : Int
    let N : Int
    let KL : Int
    let KU : Int
    let lda : Int
    let incx : Int
    let incy : Int

    let transA : TransposeType
    let order : OrderType

    var aBuf : MTLBuffer!
    var xBuf : MTLBuffer!
    var yBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        alpha = Float(params.alpha)
        beta = Float(params.beta)
        M = params.M
        N = params.N
        KL = params.KL
        KU = params.KU
        lda = params.lda
        incx = params.incx
        incy = params.incy

        order = params.order
        transA = params.transA

        metalBlas = metalBlasIn!

        let sizeA : Int = order == .ColMajor ? N * lda : M * lda
        let sizeX : Int = transA == .NoTranspose ? N * incx : M * incx
        let sizeY : Int = transA == .NoTranspose ? M * incy : N * incy

        alpha = params.alpha
        beta = params.beta

        // Can initialize as if a normal matrix, will deal with it as if it were banded. This
        // way we'll implicitly check that we aren't changing 'out-of-band' data.
        if T.self == Float.self
        {
            aArrF = []; xArrF = []; yArrF = []
            initRandomInt(&aArrF, sizeA)
            initRandomInt(&xArrF, sizeX)
            initRandomInt(&yArrF, sizeY)

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrF, M: aArrF.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrF, M: yArrF.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
        else if T.self == Double.self
        {
            aArrD = []; xArrD = []; yArrD = []
            initRandomInt(&aArrD, sizeA)
            initRandomInt(&xArrD, sizeX)
            initRandomInt(&yArrD, sizeY)
            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrD, M: aArrD.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrD, M: yArrD.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
        else if T.self == Float16.self
        {
            aArrH = []; xArrH = []; yArrH = []
            initRandomInt(&aArrH, sizeA, (0...3))
            initRandomInt(&xArrH, sizeX, (0...3))
            initRandomInt(&yArrH, sizeY, (0...3))

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrH, M: aArrH.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrH, M: yArrH.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
    }

    private func callGbmv(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSgbmv(order, transA, M, N, KL, KU, alpha, aArrF, lda, xArrF, incx, beta, &yArrF, incy) :
            T.self == Double.self ? refDgbmv(order, transA, M, N, KL, KU, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yArrD, incy) :
                                    refHgbmv(order, transA, M, N, KL, KU, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yArrH, incy)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSgbmv(order, transA, M, N, KL, KU, alpha, aBuf, lda, xBuf, incx, beta, &yBuf, incy) :
            T.self == Double.self ? metalBlas.metalDgbmv(order, transA, M, N, KL, KU, Double(alpha), aBuf, lda, xBuf, incx, Double(beta), &yBuf, incy) :
                                    metalBlas.metalHgbmv(order, transA, M, N, KL, KU, Float16(alpha), aBuf, lda, xBuf, incx, Float16(beta), &yBuf, incy)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSgbmv(order, transA, M, N, KL, KU, alpha, aArrF, lda, xArrF, incx, beta, &yArrF, incy) :
            T.self == Double.self ? metalBlas.metalDgbmv(order, transA, M, N, KL, KU, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yArrD, incy) :
                                    metalBlas.metalHgbmv(order, transA, M, N, KL, KU, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yArrH, incy)
        }
    }

    func validateGbmv() -> Bool
    {
        if T.self == Float.self
        {
            var yCpy = yArrF!
            refSgbmv(order, transA, M, N, KL, KU, alpha, aArrF, lda, xArrF, incx, beta, &yCpy, incy)
            callGbmv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrF)
            }

            return printIfNotEqual(yArrF, yCpy)
        }
        else if T.self == Double.self
        {
            var yCpy = yArrD!
            refDgbmv(order, transA, M, N, KL, KU, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yCpy, incy)
            callGbmv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrD)
            }

            return printIfNotEqual(yArrD, yCpy)
        }
        else if T.self == Float16.self
        {
            var yCpy = yArrH!
            refHgbmv(order, transA, M, N, KL, KU, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yCpy, incy)
            callGbmv(false)
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

    func benchmarkGbmv(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callGbmv(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callGbmv(benchRef)
                }
            }
        )
        return result
    }
}

class gbmvTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas2/gbmvInput"
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
        print("\tfunc,prec,order,transA,M,N,KL,KU,lda,incx,incy,alpha,beta,useBuf,coldIters,hotIters")
        print("\tgbmv", params.prec, params.order, params.transA, params.M, params.N, params.KL, params.KU, params.lda, params.incx, params.incy, params.alpha, params.beta, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func gbmvTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let gbmvFramework = GbmvFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = gbmvFramework.validateGbmv()
            XCTAssert(pass)
        }

        var result = gbmvFramework.benchmarkGbmv(false, params.coldIters, params.hotIters)
        var accResult = gbmvFramework.benchmarkGbmv(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getGbmvGflopCount(transA: params.transA, M: params.M, N: params.N, KL: params.KL, KU: params.KU) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getGbmvGflopCount(transA: params.transA, M: params.M, N: params.N, KL: params.KL, KU: params.KU) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func gbmvLauncher()
    {
        if params.prec == precisionType.fp32
        {
            gbmvTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            gbmvTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            gbmvTester(Float16(0))
        }
    }

    func testGbmv0()
    {
        for i in 0..<96
        {
            setupParams(i)
            printTestInfo("testGbmv" + String(i))
            gbmvLauncher()
        }
    }
}

