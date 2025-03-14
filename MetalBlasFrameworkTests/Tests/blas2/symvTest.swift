//
//  symvTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-03-01.
//

import XCTest
@testable import MetalBlasFramework

class SymvFramework<T: BinaryFloatingPoint>
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
    let N : Int
    let lda : Int
    let incx : Int
    let incy : Int

    let order : OrderType
    let uplo : UploType

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
        N = params.N
        lda = params.lda
        incx = params.incx
        incy = params.incy

        order = params.order
        uplo = params.uplo

        metalBlas = metalBlasIn!

        let sizeA : Int = N * lda
        let sizeX : Int = N * incx
        let sizeY : Int = N * incy

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

    private func callSymv(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSsymv(order, uplo, N, alpha, aArrF, lda, xArrF, incx, beta, &yArrF, incy) :
            T.self == Double.self ? refDsymv(order, uplo, N, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yArrD, incy) :
                                    refHsymv(order, uplo, N, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yArrH, incy)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSsymv(order, uplo, N, alpha, aBuf, lda, xBuf, incx, beta, &yBuf, incy) :
            T.self == Double.self ? metalBlas.metalDsymv(order, uplo, N, Double(alpha), aBuf, lda, xBuf, incx, Double(beta), &yBuf, incy) :
                                    metalBlas.metalHsymv(order, uplo, N, Float16(alpha), aBuf, lda, xBuf, incx, Float16(beta), &yBuf, incy)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSsymv(order, uplo, N, alpha, aArrF, lda, xArrF, incx, beta, &yArrF, incy) :
            T.self == Double.self ? metalBlas.metalDsymv(order, uplo, N, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yArrD, incy) :
                                    metalBlas.metalHsymv(order, uplo,N, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yArrH, incy)
        }
    }

    func validateSymv() -> Bool
    {
        if T.self == Float.self
        {
            var yCpy = yArrF!
            refSsymv(order, uplo, N, alpha, aArrF, lda, xArrF, incx, beta, &yCpy, incy)
            callSymv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrF)
            }

            return printIfNotEqual(yArrF, yCpy)
        }
        else if T.self == Double.self
        {
            var yCpy = yArrD!
            refDsymv(order, uplo, N, Double(alpha), aArrD, lda, xArrD, incx, Double(beta), &yCpy, incy)
            callSymv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrD)
            }

            return printIfNotEqual(yArrD, yCpy)
        }
        else if T.self == Float16.self
        {
            var yCpy = yArrH!
            refHsymv(order, uplo, N, Float16(alpha), aArrH, lda, xArrH, incx, Float16(beta), &yCpy, incy)
            callSymv(false)
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

    func benchmarkSymv(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callSymv(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callSymv(benchRef)
                }
            }
        )
        return result
    }
}

class symvTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas2/symvInput"
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
        print("\tfunc,prec,order,uplo,N,lda,incx,incy,alpha,beta,useBuf,coldIters,hotIters")
        print("\tsymv", params.prec, params.order, params.uplo,params.N, params.lda, params.incx, params.incy, params.alpha, params.beta, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func symvTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let symvFramework = SymvFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = symvFramework.validateSymv()
            XCTAssert(pass)
        }

        var result = symvFramework.benchmarkSymv(false, params.coldIters, params.hotIters)
        var accResult = symvFramework.benchmarkSymv(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getSymvGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getSymvGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func symvLauncher()
    {
        if params.prec == precisionType.fp32
        {
            symvTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            symvTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            symvTester(Float16(0))
        }
    }

    func testSymv0()
    {
        for i in 0..<50
        {
            setupParams(i)
            printTestInfo("testSymv" + String(i))
            symvLauncher()
        }
    }
}

