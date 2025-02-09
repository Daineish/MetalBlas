//
//  gemmTest.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import XCTest
@testable import MetalBlasFramework

class GemmFramework<T: BinaryFloatingPoint>
{
    var useBuffers : Bool
    var aMatH : [ Float16 ]!
    var bMatH : [ Float16 ]!
    var cMatH : [ Float16 ]!
    var aMatF : [ Float ]!
    var bMatF : [ Float ]!
    var cMatF : [ Float ]!
    var aMatD : [ Double ]!
    var bMatD : [ Double ]!
    var cMatD : [ Double ]!
    var alpha: Float
    var beta: Float
    let M : Int
    let N : Int
    let K : Int
    var lda : Int
    var ldb : Int
    var ldc : Int
    let storageMethod = MetalBlasFramework.StorageMethod.ColMajor

    let transA : TransposeType
    let transB : TransposeType

    var aBuf : MTLBuffer!
    var bBuf : MTLBuffer!
    var cBuf : MTLBuffer!

    let metalBlas: MetalBlas

    init(_ metalBlasIn: MetalBlas?, _ params: TestParams, _ useBuffersIn: Bool)
    {
        // TODO: Only supporting NN for now
        assert(T.self == Float.self || T.self == Double.self || T.self == Float16.self)

        useBuffers = useBuffersIn
        alpha = Float(params.alpha)
        M = params.M
        N = params.N
        K = params.K
        lda = params.lda
        ldb = params.ldb
        ldc = params.ldc

        transA = params.transA
        transB = params.transB

        assert(transA == .NoTranspose && transB == .NoTranspose)

        metalBlas = metalBlasIn!

        if lda < M
        {
            print("Setting lda = ", M)
            lda = M
        }
        if ldb < K
        {
            print("Setting ldb = ", K)
            ldb = K
        }
        if ldc < M
        {
            print("Setting ldc = ", M)
            ldc = M
        }

        let sizeA : Int = K * lda
        let sizeB : Int = N * ldb
        let sizeC : Int = N * ldc

        alpha = params.alpha
        beta = params.beta

        if T.self == Float.self
        {
            aMatF = []; bMatF = []; cMatF = []
            initRandomInt(&aMatF, sizeA)
            initRandomInt(&bMatF, sizeB)
            initRandomInt(&cMatF, sizeC)
            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aMatF, M: aMatF.count, [.storageModeManaged])!//.storageModePrivate)!
                bBuf = metalBlas.getDeviceBuffer(matA: bMatF, M: bMatF.count, [.storageModeManaged])!// .storageModeManaged)!
                cBuf = metalBlas.getDeviceBuffer(matA: cMatF, M: cMatF.count, [.storageModeManaged])!
            }
        }
        else if T.self == Double.self
        {
            aMatD = []; bMatD = []; cMatD = []
            initRandomInt(&aMatD, sizeA)
            initRandomInt(&bMatD, sizeB)
            initRandomInt(&cMatD, sizeC)
            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aMatD, M: aMatD.count, [.storageModeManaged])!//.storageModePrivate)!
                bBuf = metalBlas.getDeviceBuffer(matA: bMatD, M: bMatD.count, [.storageModeManaged])!// .storageModeManaged)!
                cBuf = metalBlas.getDeviceBuffer(matA: cMatD, M: cMatD.count, [.storageModeManaged])!
            }
        }
        else if T.self == Float16.self
        {
            aMatH = []; bMatH = []; cMatH = []
            initRandomInt(&aMatH, sizeA)
            initRandomInt(&bMatH, sizeB)
            initRandomInt(&cMatH, sizeC)
            if useBuffers
            {
                aBuf = metalBlas.getDeviceBuffer(matA: aMatH, M: aMatH.count, [.storageModeManaged])!//.storageModePrivate)!
                bBuf = metalBlas.getDeviceBuffer(matA: bMatH, M: bMatH.count, [.storageModeManaged])!// .storageModeManaged)!
                cBuf = metalBlas.getDeviceBuffer(matA: cMatH, M: cMatH.count, [.storageModeManaged])!
            }
        }
    }

    private func callGemm(_ callRef: Bool)
    {
        if callRef
        {
            T.self == Float.self ? refSgemm(transA, transB, M, N, K, alpha, aMatF, lda, bMatF, ldb, beta, &cMatF, ldc) :
            T.self == Double.self ? refDgemm(transA, transB, M, N, K, Double(alpha), aMatD, lda, bMatD, ldb, Double(beta), &cMatD, ldc) :
                                    refHgemm(transA, transB, M, N, K, Float16(alpha), aMatH, lda, bMatH, ldb, Float16(beta), &cMatH, ldc)
        }
        else if useBuffers
        {
            T.self == Float.self ? metalBlas.metalSgemm(storageMethod, transA, transB, M, N, K, alpha, aBuf, lda, bBuf, ldb, beta, &cBuf, ldc) :
            T.self == Double.self ? metalBlas.metalDgemm(storageMethod, transA, transB, M, N, K, Double(alpha), aBuf, lda, bBuf, ldb, Double(beta), &cBuf, ldc) :
                                    metalBlas.metalHgemm(storageMethod, transA, transB, M, N, K, Float16(alpha), aBuf, lda, bBuf, ldb, Float16(beta), &cBuf, ldc)
        }
        else
        {
            T.self == Float.self ? metalBlas.metalSgemm(storageMethod, transA, transB, M, N, K, alpha, aMatF, lda, bMatF, ldb, beta, &cMatF, ldc) :
            T.self == Double.self ? metalBlas.metalDgemm(storageMethod, transA, transB, M, N, K, Double(alpha), aMatD, lda, bMatD, ldb, Double(beta), &cMatD, ldc) :
                                    metalBlas.metalHgemm(storageMethod, transA, transB, M, N, K, Float16(alpha), aMatH, lda, bMatH, ldb, Float16(beta), &cMatH, ldc)
        }
    }

    func validateGemm() -> Bool
    {
        if T.self == Float.self
        {
            var cCpy = cMatF!
            refSgemm(transA, transB, M, N, K, alpha, aMatF, lda, bMatF, ldb, beta, &cCpy, ldc)
            callGemm(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(cBuf, &cMatF)
            }

            return printIfNotEqual(cMatF, cCpy)
        }
        else if T.self == Double.self
        {
            var cCpy = cMatD!
            refDgemm(transA, transB, M, N, K, Double(alpha), aMatD, lda, bMatD, ldb, Double(beta), &cCpy, ldc)
            callGemm(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(cBuf, &cMatD)
            }

            return printIfNotEqual(cMatD, cCpy)
        }
        else if T.self == Float16.self
        {
            var cCpy = cMatH!
            refHgemm(transA, transB, M, N, K, Float16(alpha), aMatH, lda, bMatH, ldb, Float16(beta), &cCpy, ldc)
            callGemm(false)
            if useBuffers
            {
                metalBlas.copyBufToArray(cBuf, &cMatH)
            }

            return printIfNotEqual(cMatH, cCpy)
        }
        else
        {
            print("Error: Unsupported precision")
            return false
        }
    }

    func benchmarkGemm(_ benchRef: Bool, _ coldIters: Int, _ hotIters: Int) -> Duration
    {
        for _ in 0...coldIters {
            callGemm(benchRef)
        }

        let clock = ContinuousClock()
        let result = clock.measure(
            {
                for _ in 0...hotIters {
                    callGemm(benchRef)
                }
            }
        )
        return result
    }
}

class gemmTestMetal: XCTestCase
{
    var params : TestParams!
    let fileName = "Projects/CodingProjects/Swift Projects/MetalBlas/MetalBlasFrameworkTests/Data/blas3/gemmInput"
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
        print("\tfunc,prec,transA,transB,M,N,K,alpha,beta,lda,ldb,ldc,useBuf,coldIters,hotIters")
        print("\tgemm", params.prec, params.transA, params.transB, params.M, params.N, params.K, params.alpha, params.beta, params.lda, params.ldb, params.ldc, params.useBuffers, params.coldIters, params.hotIters, separator: ",")
    }

    func gemmTester<T: BinaryFloatingPoint>(_: T)
    {
        let benchAccelerate = true
        let verify = true

        let gemmFramework = GemmFramework<T>(metalBlas, params, useBuffersDirectly)
        
        if verify
        {
            let pass = gemmFramework.validateGemm()
            XCTAssert(pass)
        }

        var result = gemmFramework.benchmarkGemm(false, params.coldIters, params.hotIters)
        var accResult = gemmFramework.benchmarkGemm(true, params.coldIters, params.hotIters)
        result /= params.hotIters
        accResult /= params.hotIters

        let avgTime = Double(result.components.attoseconds) / 1e18
        let gflops = getGemmGflopCount(M: params.M, N: params.N, K: params.K) / avgTime
        print("MetalBlas time in s: ", avgTime / 1e18)
        print("MetalBlas avg gflops: ", gflops)

        if benchAccelerate
        {
            let accAvgTime = Double(accResult.components.attoseconds) / 1e18
            let accgflops = getGemmGflopCount(M: params.M, N: params.N, K: params.K) / accAvgTime
            print("Accelerate time in s: ", accAvgTime / 1e18)
            print("Accelerate avg gflops: ", accgflops)
        }
    }

    func gemmLauncher()
    {
        if params.prec == .fp32
        {
            gemmTester(Float(0))
        }
        else if params.prec == .fp64
        {
            gemmTester(Double(0))
        }
        else if params.prec == .fp16
        {
            gemmTester(Float16(0))
        }
    }

    func testGemm0()
    {
        setupParams(0)
        printTestInfo("testGemm0")
        gemmLauncher()
    }

    func testGemm1()
    {
        setupParams(1)
        printTestInfo("testGemm1")
        gemmLauncher()
    }

    func testGemm2()
    {
        setupParams(2)
        printTestInfo("testGemm2")
        gemmLauncher()
    }

    func testGemm3()
    {
        setupParams(3)
        printTestInfo("testGemm3")
        gemmLauncher()
    }
}

