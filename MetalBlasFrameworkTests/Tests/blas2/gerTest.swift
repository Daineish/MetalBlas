//
//  gerTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

import XCTest
@testable import MetalBlasFramework

class GerFramework<T: BinaryFloatingPoint>
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
    let M : Int
    let N : Int
    let lda : Int
    let incx : Int
    let incy : Int

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
        M = params.M
        N = params.N
        lda = params.lda
        incx = params.incx
        incy = params.incy

        order = params.order

        metalBlas = metalBlasIn!

        let sizeA : Int = order == .ColMajor ? N * lda : M * lda
        let sizeX : Int = M * incx
        let sizeY : Int = N * incy

        alpha = params.alpha

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

    private func callGer(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSger(order, M, N, alpha, xArrF, incx, yArrF, incy, &aArrF, lda) :
            T.self == Double.self ? refDger(order, M, N, Double(alpha), xArrD, incx, yArrD, incy, &aArrD, lda) :
                                    refHger(order, M, N, Float16(alpha), xArrH, incx, yArrH, incy, &aArrH, lda)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSger(order, M, N, alpha, xBuf, incx, yBuf, incy, &aBuf, lda) :
            T.self == Double.self ? metalBlas.metalDger(order, M, N, Double(alpha), xBuf, incx, yBuf, incy, &aBuf, lda) :
                                    metalBlas.metalHger(order, M, N, Float16(alpha), xBuf, incx, yBuf, incy, &aBuf, lda)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSger(order, M, N, alpha, xArrF, incx, yArrF, incy, &aArrF, lda) :
            T.self == Double.self ? metalBlas.metalDger(order, M, N, Double(alpha), xArrD, incx, yArrD, incy, &aArrD, lda) :
                                    metalBlas.metalHger(order, M, N, Float16(alpha), xArrH, incx, yArrH, incy, &aArrH, lda)
        }
    }

    func validateGer() -> Bool
    {
        if T.self == Float.self
        {
            var aCpy = aArrF!
            refSger(order, M, N, alpha, xArrF, incx, yArrF, incy, &aCpy, lda)
            callGer(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrF)
            }

            return printIfNotEqual(aArrF, aCpy)
        }
        else if T.self == Double.self
        {
            var aCpy = aArrD!
            refDger(order, M, N, Double(alpha), xArrD, incx, yArrD, incy, &aCpy, lda)
            callGer(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrD)
            }

            return printIfNotEqual(aArrD, aCpy)
        }
        else if T.self == Float16.self
        {
            var aCpy = aArrH!
            refHger(order, M, N, Float16(alpha), xArrH, incx, yArrH, incy, &aCpy, lda)
            callGer(false)
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

    func benchmarkGer(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callGer(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callGer(benchRef)
                }
            }
        )
        return result
    }
}

class gerTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/gerInput"
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
        print("\tfunc,prec,order,M,N,lda,incx,incy,alpha,beta,useBuf,coldIters,hotIters")
        print("\tger", params.prec, params.order, params.M, params.N, params.lda, params.incx, params.incy, params.alpha, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func gerTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let gerFramework = GerFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = gerFramework.validateGer()
            XCTAssert(pass)
        }

        var result = gerFramework.benchmarkGer(false, params.coldIters, params.hotIters)
        var accResult = gerFramework.benchmarkGer(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getGerGflopCount(M: params.M, N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getGerGflopCount(M: params.M, N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func gerLauncher()
    {
        if params.prec == precisionType.fp32
        {
            gerTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            gerTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            gerTester(Float16(0))
        }
    }

    func testGer0()
    {
        for i in 0..<8
        {
            setupParams(i)
            printTestInfo("testGer" + String(i))
            gerLauncher()
        }
    }
}

