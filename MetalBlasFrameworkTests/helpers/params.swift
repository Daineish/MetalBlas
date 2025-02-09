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
    var KL: Int
    var KU: Int

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
        N = 1
        K = 1
        KL = 0
        KU = 0
        lda = 5
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
        KL = cpy.KL
        KU = cpy.KU
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

    func parsePrec(s: String) -> precisionType
    {
        if(s == "s" || s == "fp32")
        {
            return precisionType.fp32
        }
        else if(s == "h" || s == "fp16")
        {
            return precisionType.fp16
        }
        else if(s == "d" || s == "fp64")
        {
            return precisionType.fp64
        }
        return precisionType.fp32
    }

    func parseFunc(s: String) -> funcType
    {
        if(s == "axpy")
        {
            return funcType.axpy
        }
        else if(s == "scal")
        {
            return funcType.scal
        }
        else if(s == "copy")
        {
            return funcType.copy
        }
        else if(s == "gemv")
        {
            return funcType.gemv
        }
        else if(s == "gemm")
        {
            return funcType.gemm
        }
        else if(s == "asum")
        {
            return funcType.asum
        }
        return funcType.axpy
    }

    func parseOrder(s: String) -> OrderType
    {
        if s == "R" || s == "Row" || s == "RowMajor"
        {
            return .RowMajor
        }
        return .ColMajor
    }

    func parseTrans(s: String) -> TransposeType
    {
        if(s == "T" || s == "t" || s == "Transpose")
        {
            return .Transpose
        }
        else if(s == "N" || s == "n" || s == "NoTranspose")
        {
            return .NoTranspose
        }
        else if(s == "C" || s == "c" || s == "ConjTranspose")
        {
            return .ConjTranspose
        }
        return .NoTranspose
    }

    mutating func set(_ header: String, _ value: String)
    {
        switch header
        {
        case "func", "function":
            self.function = parseFunc(s: value)
        case "prec":
            self.prec = parsePrec(s: value)
        case "M", "m":
            self.M = Int(value) ?? 0
        case "N", "n":
            self.N = Int(value) ?? 0
        case "K", "k":
            self.K = Int(value) ?? 0
        case "KL", "kl":
            self.KL = Int(value) ?? 0
        case "KU", "ku":
            self.KU = Int(value) ?? 0
        case "incx":
            self.incx = Int(value) ?? 0
        case "incy":
            self.incy = Int(value) ?? 0
        case "lda":
            self.lda = Int(value) ?? 0
        case "ldb":
            self.ldb = Int(value) ?? 0
        case "ldc":
            self.ldc = Int(value) ?? 0
        case "ldd":
            self.ldd = Int(value) ?? 0
        case "alpha", "ɑ":
            self.alpha = Float(value) ?? 1.0
        case "beta", "β":
            self.beta = Float(value) ?? 0.0
        case "order":
            self.order = parseOrder(s: value)
        case "transA", "transposeA":
            self.transA = parseTrans(s: value)
        case "transB", "transposeB":
            self.transB = parseTrans(s: value)
        case "useBuf":
            self.useBuffers = Bool(value) ?? false
        case "coldIters", "j":
            self.coldIters = Int(value) ?? 2
        case "hotIters", "i":
            self.hotIters = Int(value) ?? 10
        case "notes", "Notes":
            return // do nothing
        default:
            print("Input param ", header, " ignored.")
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
