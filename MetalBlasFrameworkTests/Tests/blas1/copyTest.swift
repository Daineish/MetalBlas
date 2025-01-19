//
//  copyTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

import XCTest
@testable import MetalBlasFramework

class CopyFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var xArrH : [ Float16 ]!
    var yArrH : [ Float16 ]!
    var xArrF : [ Float ]!
    var yArrF : [ Float ]!
    var xArrD : [ Double ]!
    var yArrD : [ Double ]!
    let N : Int
    let incx : Int
    let incy : Int

    var xBuf : MTLBuffer!
    var yBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        N = params.N
        incx = params.incx
        incy = params.incy

        metalBlas = metalBlasIn!

        let sizeX : Int = params.N * params.incx
        let sizeY : Int = params.N * params.incy

        if T.self == Float.self
        {
            xArrF = []; yArrF = []
            initRandom(&xArrF, sizeX)
            initRandom(&yArrF, sizeY)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrF, M: xArrF.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrF, M: yArrF.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
        else if T.self == Double.self
        {
            xArrD = []; yArrD = []
            initRandom(&xArrD, sizeX)
            initRandom(&yArrD, sizeY)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrD, M: xArrD.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrD, M: yArrD.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
        else if T.self == Float16.self
        {
            xArrH = []; yArrH = []
            initRandom(&xArrH, sizeX)
            initRandom(&yArrH, sizeY)
            if useBuffers
            {
                xBuf = metalBlas.getDeviceBuffer(matA: xArrH, M: xArrH.count, [.storageModeManaged])!//.storageModePrivate)!
                yBuf = metalBlas.getDeviceBuffer(matA: yArrH, M: yArrH.count, [.storageModeManaged])!// .storageModeManaged)!
            }
        }
    }

    private func callCopy(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refScopy(N, xArrF, incx, &yArrF, incy) :
            T.self == Double.self ? refDcopy(N, xArrD, incx, &yArrD, incy) :
                                    refHcopy(N, xArrH, incx, &yArrH, incy)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalScopy(N, xBuf, incx, &yBuf, incy) :
            T.self == Double.self ? metalBlas.metalDcopy(N, xBuf, incx, &yBuf, incy) :
                                    metalBlas.metalHcopy(N, xBuf, incx, &yBuf, incy)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalScopy(N, xArrF, incx, &yArrF, incy) :
            T.self == Double.self ? metalBlas.metalDcopy(N, xArrD, incx, &yArrD, incy) :
                                    metalBlas.metalHcopy(N, xArrH, incx, &yArrH, incy)
        }
    }

    func validateCopy() -> Bool
    {
        if T.self == Float.self
        {
            var yCpy = yArrF!
            refScopy(N, xArrF, incx, &yCpy, incy)
            callCopy(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrF)
            }

            return printIfNotEqual(yArrF, yCpy)
        }
        else if T.self == Double.self
        {
            var yCpy = yArrD!
            refDcopy(N, yArrD, incx, &yCpy, incy)
            callCopy(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(yBuf, &yArrD)
            }

            return printIfNotEqual(yArrD, yCpy)
        }
        else if T.self == Float16.self
        {
            var yCpy = yArrH!
            refHcopy(N, xArrH, incx, &yCpy, incy)
            callCopy(false)
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

    func benchmarkCopy(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callCopy(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callCopy(benchRef)
                }
            }
        )
        return result
    }
}

class copyTest: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/copyInput"
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
        print("\tfunc,prec,N,incx,incy,useBuf,coldIters,hotIters")
        print("\tcopy", params.prec, params.N, params.incx, params.incy, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func copyTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let copyFramework = CopyFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = copyFramework.validateCopy()
            XCTAssert(pass)
        }

        var result = copyFramework.benchmarkCopy(false, params.coldIters, params.hotIters)
        var accResult = copyFramework.benchmarkCopy(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gbytes = getCopyGbyteCount(N: params.N) * Double(MemoryLayout<T>.stride) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gbytes: ", gbytes)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgbytes = getCopyGbyteCount(N: params.N) * Double(MemoryLayout<T>.stride) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gbytes: ", accgbytes)
        }
    }

    func copyLauncher()
    {
        if params.prec == precisionType.fp32
        {
            copyTester(Float(0))
        }
        else if params.prec == precisionType.fp64
        {
            copyTester(Double(0))
        }
        else if params.prec == precisionType.fp16
        {
            copyTester(Float16(0))
        }
    }

    func testCopy0()
    {
        // TODO: test more than one line per test? Probably on a per-file basis or something
        setupParams(0)
        printTestInfo("testCopy0")
        copyLauncher()
    }

    func testCopy1()
    {
        setupParams(1)
        printTestInfo("testCopy1")
        copyLauncher()
    }

    func testCopy2()
    {
        setupParams(2)
        printTestInfo("testCopy2")
        copyLauncher()
    }

    func testCopy3()
    {
        setupParams(3)
        printTestInfo("testCopy3")
        copyLauncher()
    }
}

