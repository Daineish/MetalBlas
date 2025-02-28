//
//  spr2Test.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-22.
//

import XCTest
@testable import MetalBlasFramework

class Spr2Framework<T: BinaryFloatingPoint>
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
    let N : Int
    let incx : Int
    let incy: Int

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
        N = params.N
        incx = params.incx
        incy = params.incy

        order = params.order
        uplo = params.uplo

        metalBlas = metalBlasIn!

        let sizeA : Int = N * (N + 1) / 2
        let sizeX : Int = N * incx
        let sizeY : Int = N * incy

        alpha = params.alpha

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
                yBuf = metalBlas.getDeviceBuffer(matA: yArrF, M: yArrF.count, [.storageModeManaged])!
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
                yBuf = metalBlas.getDeviceBuffer(matA: yArrD, M: yArrD.count, [.storageModeManaged])!
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
                yBuf = metalBlas.getDeviceBuffer(matA: yArrH, M: yArrH.count, [.storageModeManaged])!
            }
        }
    }

    private func callSpr2(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSspr2(order, uplo, N, alpha, xArrF, incx, yArrF, incy, &aArrF) :
            T.self == Double.self ? refDspr2(order, uplo, N, Double(alpha), xArrD, incx, yArrD, incy, &aArrD) :
                                    refHspr2(order, uplo, N, Float16(alpha), xArrH, incx, yArrH, incy, &aArrH)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSspr2(order, uplo, N, alpha, xBuf, incx, yBuf, incy, &aBuf) :
            T.self == Double.self ? metalBlas.metalDspr2(order, uplo, N, Double(alpha), xBuf, incx, yBuf, incy, &aBuf) :
                                    metalBlas.metalHspr2(order, uplo, N, Float16(alpha), xBuf, incx, yBuf, incy, &aBuf)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSspr2(order, uplo, N, alpha, xArrF, incx, yArrF, incy, &aArrF) :
            T.self == Double.self ? metalBlas.metalDspr2(order, uplo, N, Double(alpha), xArrD, incx, yArrD, incy, &aArrD) :
                                    metalBlas.metalHspr2(order, uplo,N, Float16(alpha), xArrH, incx, yArrH, incy, &aArrH)
        }
    }

    func validateSpr2() -> Bool
    {
        if T.self == Float.self
        {
            var aCpy = aArrF!
            refSspr2(order, uplo, N, alpha, xArrF, incx, yArrF, incy, &aCpy)
            callSpr2(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrF)
            }

            return printIfNotEqual(aArrF, aCpy)
        }
        else if T.self == Double.self
        {
            var aCpy = aArrD!
            refDspr2(order, uplo, N, Double(alpha), xArrD, incx, yArrD, incy, &aCpy)
            callSpr2(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(aBuf, &aArrD)
            }

            return printIfNotEqual(aArrD, aCpy)
        }
        else if T.self == Float16.self
        {
            var aCpy = aArrH!
            refHspr2(order, uplo, N, Float16(alpha), xArrH, incx, yArrH, incy, &aCpy)
            callSpr2(false)
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

    func benchmarkSpr2(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callSpr2(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callSpr2(benchRef)
                }
            }
        )
        return result
    }
}

class spr2Test: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas2/spr2Input"
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
        print("\tfunc,prec,order,uplo,N,incx,incy,alpha,useBuf,coldIters,hotIters")
        print("\tspr2", params.prec, params.order, params.uplo, params.N, params.incx, params.incy, params.alpha, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func spr2Tester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let spr2Framework = Spr2Framework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = spr2Framework.validateSpr2()
            XCTAssert(pass)
        }

        var result = spr2Framework.benchmarkSpr2(false, params.coldIters, params.hotIters)
        var accResult = spr2Framework.benchmarkSpr2(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getSpr2GflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getSpr2GflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func spr2Launcher()
    {
        if params.prec == precisionType.fp32
        {
            spr2Tester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            spr2Tester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            spr2Tester(Float16(0))
        }
    }

    func testSpr20()
    {
        for i in 0..<64
        {
            setupParams(i)
            printTestInfo("testSpr2" + String(i))
            spr2Launcher()
        }
    }
}

