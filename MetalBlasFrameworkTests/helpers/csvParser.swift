//
//  csvParser.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

import Foundation

class csvParser
{
    var filename : URL

//    init()
//    {
//        filename = "data.csv"
//    }
//
//    init(file: String)
//    {
//        filename = file
//    }

    init(file: URL)
    {
        filename = file
    }

    func readFile() -> String
    {
        var data : String
        do
        {
            data = try String(contentsOf: filename, encoding: .utf8)
        }
        catch
        {
            print(error)
            data = ""
        }

        return data
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

    func parse() -> [TestParams]
    {
        var allParams : [TestParams] = []
        let data = readFile()

        var rows = data.components(separatedBy: "\n")
        rows.removeFirst()

        for row in rows
        {
            if row.isEmpty
            {
                break
            }

            let cols = row.components(separatedBy: ",")
            var param = TestParams()
            param.function = parseFunc(s:cols[0])
            param.prec = parsePrec(s: cols[1])
            param.M = Int(cols[2]) ?? 0
            param.N = Int(cols[3]) ?? 0
            param.K = Int(cols[4]) ?? 0
            param.incx = Int(cols[5]) ?? 0
            param.incy = Int(cols[6]) ?? 0
            param.lda = Int(cols[7]) ?? 0
            param.ldb = Int(cols[8]) ?? 0
            param.ldc = Int(cols[9]) ?? 0
            param.ldd = Int(cols[10]) ?? 0
            param.alpha = Float(cols[11]) ?? 2.0
            param.beta = Float(cols[12]) ?? 1.0
            param.transA = parseTrans(s: cols[13])
            param.transB = parseTrans(s: cols[14])
            param.useBuffers = Bool(cols[15]) ?? false
            param.coldIters = Int(cols[16]) ?? 2
            param.hotIters = Int(cols[17]) ?? 10

            allParams.append(param)
        }

        return allParams
    }
}
