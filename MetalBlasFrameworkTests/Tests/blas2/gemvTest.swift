//
//  gemvTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

import XCTest
@testable import MetalBlasFramework

class GemvFramework<T: BinaryFloatingPoint>
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

    private func callGemv(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSgemv(order, transA, M, N, alpha, aArrF, lda, xArrF, incx, beta, &yArrF, incy) :
            T.self == Double.self ? refDgemv(order, transA, M, N, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yArrD, incy) :
                                    refHgemv(order, transA, M, N, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yArrH, incy)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSgemv(order, transA, M, N, alpha, aBuf, lda, xBuf, incx, beta, &yBuf, incy) :
            T.self == Double.self ? metalBlas.metalDgemv(order, transA, M, N, Double(alpha), aBuf, lda, xBuf, incx, Double(beta), &yBuf, incy) :
                                    metalBlas.metalHgemv(order, transA, M, N, Float16(alpha), aBuf, lda, xBuf, incx, Float16(beta), &yBuf, incy)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSgemv(order, transA, M, N, alpha, aArrF, lda, xArrF, incx, beta, &yArrF, incy) :
            T.self == Double.self ? metalBlas.metalDgemv(order, transA, M, N, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yArrD, incy) :
                                    metalBlas.metalHgemv(order, transA, M, N, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yArrH, incy)
        }
    }

    func validateGemv() -> Bool
    {
        if T.self == Float.self
        {
            var yCpy = yArrF!
            refSgemv(order, transA, M, N, alpha, aArrF, lda, xArrF, incx, beta, &yCpy, incy)
            callGemv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrF)
            }

            return printIfNotEqual(yArrF, yCpy)
//            return printIfNotNear(yArrF, yCpy, transA == .NoTranspose ? 10 * N : 10 * M)
        }
        else if T.self == Double.self
        {
            var yCpy = yArrD!
            refDgemv(order, transA, M, N, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yCpy, incy)
            callGemv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrD)
            }

            return printIfNotEqual(yArrD, yCpy)
//            return printIfNotNear(yArrD, yCpy, transA == .NoTranspose ? 10 * N : 10 * M)
        }
        else if T.self == Float16.self
        {
            var yCpy = yArrH!
            refHgemv(order, transA, M, N, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yCpy, incy)
            callGemv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrH)
            }

            return printIfNotEqual(yArrH, yCpy)
//            return printIfNotNear(yArrH, yCpy, transA == .NoTranspose ? 10 * N : 10 * M)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkGemv(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callGemv(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callGemv(benchRef)
                }
            }
        )
        return result
    }
}

class gemvTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/gemvInput"
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
        print("\tfunc,prec,order,transA,M,N,lda,incx,incy,alpha,beta,useBuf,coldIters,hotIters")
        print("\tgemv", params.prec, params.order, params.transA, params.M, params.N, params.lda, params.incx, params.incy, params.alpha, params.beta, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func gemvTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let gemvFramework = GemvFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = gemvFramework.validateGemv()
            XCTAssert(pass)
        }

        var result = gemvFramework.benchmarkGemv(false, params.coldIters, params.hotIters)
        var accResult = gemvFramework.benchmarkGemv(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getGemvGflopCount(transA: params.transA, M: params.M, N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getGemvGflopCount(transA: params.transA, M: params.M, N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func gemvLauncher()
    {
        if params.prec == precisionType.fp32
        {
            gemvTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            gemvTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            gemvTester(Float16(0))
        }
    }

    func testGemv0()
    {
        for i in 0..<8
        {
            setupParams(i)
            printTestInfo("testGemv" + String(i))
            gemvLauncher()
        }
    }
}

