//
//  params.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Foundation
import MetalBlasFramework

enum precisionType
{
    case fp64
    case fp32
    case fp16
    case bf16
}

enum funcType
{
    case axpy
    case scal
    case copy
    case gemv
    case gemm
    case asum
}

public typealias OrderType = MetalBlasFramework.OrderType
public typealias TransposeType = MetalBlasFramework.TransposeType

struct TestParams
{
    var M: Int
    var N: Int
    var K: Int

    var lda: Int
    var ldb: Int
    var ldc: Int
    var ldd: Int

    var incx: Int
    var incy: Int

    var alpha: Float
    var beta: Float

    var order: OrderType

    var transA : TransposeType
    var transB : TransposeType

    var coldIters = 2
    var hotIters = 100

    var function: funcType
    var prec: precisionType

    var useBuffers: Bool

    init()
    {
        M = 1
        N = 32768
        K = 1
        lda = 1
        ldb = 1
        ldc = 1
        ldd = 1
        incx = 1
        incy = 1

        alpha = 1.0
        beta = 0.0

        order = .ColMajor

        transA = .NoTranspose
        transB = .NoTranspose

        function = funcType.axpy
        prec = precisionType.fp32
        useBuffers = false
    }

    init(cpy : TestParams)
    {
        M = cpy.M
        N = cpy.N
        K = cpy.K
        lda = cpy.lda
        ldb = cpy.ldb
        ldc = cpy.ldc
        ldd = cpy.ldd
        incx = cpy.incx
        incy = cpy.incy
        alpha = cpy.alpha
        beta = cpy.beta
        order = cpy.order
        transA = cpy.transA
        transB = cpy.transB
        function = cpy.function
        prec = cpy.prec
        
        coldIters = cpy.coldIters
        hotIters = cpy.hotIters

        useBuffers = cpy.useBuffers
    }

    init(filename: URL, lineNum: Int)
    {
        self.init()
        if !globalTestParams.isEmpty
        {
            self.init(cpy: globalTestParams[lineNum])
        }
        else
        {
            let csvParser = csvParser(file: filename)
            globalTestParams = csvParser.parse()

            self.init(cpy: globalTestParams[lineNum])
        }
    }
}

func getParams(fileName: String, i: Int) -> TestParams
{
    let dir = try? FileManager.default.url(for: .documentDirectory,
          in: .userDomainMask, appropriateFor: nil, create: true)

    // TODO: get lower lines to work instead of this
    guard let filepath = dir?.appendingPathComponent(fileName).appendingPathExtension("csv") else {
        fatalError("Not able to create URL")
    }
    
//        guard let filepath = Bundle.main.url(forResource: "input", withExtension: "csv") else {
//            fatalError("Not able to find data file")
//        }
    return TestParams(filename: filepath, lineNum: i)
}

var globalTestParams : [TestParams] = []
