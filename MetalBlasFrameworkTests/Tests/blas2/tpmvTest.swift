//
//  tpmvTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-06-21.
//

import XCTest
@testable import MetalBlasFramework

class TpmvFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var aArrH : [ Float16 ]!
    var xArrH : [ Float16 ]!
    var aArrF : [ Float ]!
    var xArrF : [ Float ]!
    var aArrD : [ Double ]!
    var xArrD : [ Double ]!
    let N : Int
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
        incx = params.incx

        order = params.order
        uplo = params.uplo
        trans = params.transA
        diag = params.diag

        metalBlas = metalBlasIn!

        let sizeA : Int = (N * (N + 1)) / 2

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
            initRandomVec(&aArrH, sizeA, 1)
            initRandomVec(&xArrH, N, incx, (0...3))

            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aArrH, M: aArrH.count, [.storageModeManaged])!
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
            }
        }
    }

    private func callTpmv(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refStpmv(order, uplo, trans, diag,  N, aArrF, &xArrF, incx) :
            T.self == Double.self ? refDtpmv(order, uplo, trans, diag,  N, aArrD, &xArrD, incx) :
                                    refHtpmv(order, uplo, trans, diag,  N, aArrH, &xArrH, incx)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalStpmv(order, uplo, trans, diag,  N, aBuf, &xBuf, incx) :
            T.self == Double.self ? metalBlas.metalDtpmv(order, uplo, trans, diag,  N, aBuf, &xBuf, incx) :
                                    metalBlas.metalHtpmv(order, uplo, trans, diag,  N, aBuf, &xBuf, incx)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalStpmv(order, uplo, trans, diag, N, aArrF, &xArrF, incx) :
            T.self == Double.self ? metalBlas.metalDtpmv(order, uplo, trans, diag, N, aArrD, &xArrD, incx) :
                                    metalBlas.metalHtpmv(order, uplo, trans, diag, N, aArrH, &xArrH, incx)
        }
    }

    func validateTpmv() -> Bool
    {
        if T.self == Float.self
        {
            var xCpy = xArrF!
            refStpmv(order, uplo, trans, diag, N, aArrF, &xCpy, incx)
            callTpmv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrF)
            }

            return printIfNotEqual(xArrF, xCpy)
        }
        else if T.self == Double.self
        {
            var xCpy = xArrD!
            refDtpmv(order, uplo, trans, diag, N, aArrD, &xCpy, incx)
            callTpmv(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(xBuf, &xArrD)
            }

            return printIfNotEqual(xArrD, xCpy)
        }
        else if T.self == Float16.self
        {
            var xCpy = xArrH!
            refHtpmv(order, uplo, trans, diag, N, aArrH, &xCpy, incx)
            callTpmv(false)
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

    func benchmarkTpmv(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callTpmv(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callTpmv(benchRef)
                }
            }
        )
        return result
    }
}

class tpmvTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas2/tpmvInput"
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
        print("\tfunc,prec,order,uplo,trans,diag,N,incx,useBuf,coldIters,hotIters")
        print("\ttpmv", params.prec, params.order, params.uplo,params.transA,params.diag,params.N,params.incx,params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func tpmvTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let tpmvFramework = TpmvFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = tpmvFramework.validateTpmv()
            XCTAssert(pass)
        }

        var result = tpmvFramework.benchmarkTpmv(false, params.coldIters, params.hotIters)
        var accResult = tpmvFramework.benchmarkTpmv(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getTpmvGflopCount(N: params.N) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getTrmvGflopCount(N: params.N) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func tpmvLauncher()
    {
        if params.prec == precisionType.fp32
        {
            tpmvTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            tpmvTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            tpmvTester(Float16(0))
        }
    }

    func testTpmv0()
    {
        for i in 0..<128
        {
            setupParams(i)
            printTestInfo("testTpmv" + String(i))
            tpmvLauncher()
        }
    }
}

