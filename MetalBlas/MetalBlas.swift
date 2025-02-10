//
//  MetalBlas.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-04.
//

import Metal

public enum OrderType: UInt32
{
    case ColMajor  = 0
    case RowMajor  = 1
}

public enum TransposeType: UInt32
{
    case NoTranspose
    case Transpose
    case ConjTranspose
}

public enum UploType: UInt32
{
    case FillUpper
    case FillLower
}

public enum StorageMethod
{
    case ColMajor
    case RowMajor
}

public func reinterpret<T, U>(input: [T]) -> [U]
{
    guard MemoryLayout<T>.stride == MemoryLayout<U>.stride else {
        fatalError("Incompatible types")
    }

    return input.withUnsafeBufferPointer
    {
        buffer in
            return buffer.baseAddress!.withMemoryRebound(to: U.self, capacity: input.count)
            {
                Array(UnsafeBufferPointer(start: $0, count: input.count))
            }
    }
}

public class MetalBlas
{
    internal let device: MTLDevice
    internal let library: MTLLibrary
    internal var commandQueue: MTLCommandQueue

    // Map of API name -> pipeline state
    internal let pipelineStates: [String: MTLComputePipelineState]

    public init?()
    {
        // TODO: store buffers and stuff in initialization so we don't
        //       have to realize that cost when calling the functions (?)
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device.")
            return nil
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue.")
        }
        self.commandQueue = commandQueue

        // TODO: why can't I get the default library to work -.-
        guard let url = URL(string: "/Users/dainemcniven/Library/Developer/Xcode/DerivedData/MetalBlas-aeiwiyrueoewwldvqfcjuimqsrla/Build/Products/Debug/MetalBlasFramework.framework/Versions/A/Resources/default.metallib") else { return nil }
        do {
            // self.library = device.makeDefaultLibrary()
            self.library = try device.makeLibrary(URL: url)
        } catch {
            fatalError("Failed to create Metal library")
        }

        do
        {
            pipelineStates = [
                // Level 1
                "hasum": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHasum")!),
                "sasum": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSasum")!),
                "haxpy": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHaxpy")!),
                "saxpy": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSaxpy")!),
                "hcopy": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHcopy")!),
                "scopy": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalScopy")!),
                "hdot": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHdot")!),
                "sdot": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSdot")!),
                "ihamin": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalIhamin")!),
                "isamin": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalIsamin")!),
                "ihamax": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalIhamax")!),
                "isamax": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalIsamax")!),
                "hnrm2": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHnrm2")!),
                "snrm2": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSnrm2")!),
                "hrot": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHrot")!),
                "srot": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSrot")!),
                "hrotm": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHrotm")!),
                "srotm": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSrotm")!),
                "hrotmg": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHrotmg")!),
                "srotmg": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSrotmg")!),
                "hrotg": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHrotg")!),
                "srotg": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSrotg")!),
                "hscal": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHscal")!),
                "sscal": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSscal")!),
                "hswap": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHswap")!),
                "sswap": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSswap")!),

                // Level 2
                "hgbmv": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHgbmv")!),
                "sgbmv": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSgbmv")!),
                "hgemv": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHgemv")!),
                "sgemv": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSgemv")!),
                "hger": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHger")!),
                "sger": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSger")!),
                "hsbmv": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHsbmv")!),
                "ssbmv": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSsbmv")!),

                // Level 3
                "hgemm": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHgemm")!),
                "sgemm": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSgemm")!),

                // Helpers
                "sreduce": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalsReduce")!),
                "hreduce": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalhReduce")!),
                "sreduceMax": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalsReduceMax")!),
                "hreduceMax": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalhReduceMax")!),
                "sreduceMin": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalsReduceMin")!),
                "hreduceMin": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalhReduceMin")!),
                "sreducesq": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalsReduceSq")!),
                "hreducesq": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalhReduceSq")!)
                
            ]
        }
        catch {
            fatalError("Failed to create pipeline states.")
        }
    }

    public func createCommandQueue() -> MTLCommandQueue?
    {
        return device.makeCommandQueue()
    }

    public func createFunction(named name: String) -> MTLFunction?
    {
        return library.makeFunction(name: name)
    }

    public func getDeviceBuffer<T: Numeric>(matA: [T], M: Int, _ ops: MTLResourceOptions) -> MTLBuffer?
    {
        device.makeBuffer(bytes: matA, length: matA.count * MemoryLayout<T>.stride, options: ops)
    }

    public func getDeviceBuffer<T: Numeric>(_ a: inout T, _ ops: MTLResourceOptions) -> MTLBuffer?
    {
        // TODO: ?
        device.makeBuffer(bytes: &a, length: MemoryLayout<T>.stride, options: ops)
    }

    public func copyBufToArray<T: Numeric>(_ bufA: MTLBuffer, _ matA: inout [T])
    {
        // Copy the data into the array
        let size = bufA.length / MemoryLayout<T>.stride
        let bufferPointer = bufA.contents()
        let typePtr = bufferPointer.bindMemory(to: T.self, capacity: size)
        let typeBuffer = UnsafeBufferPointer(start: typePtr, count: size)
        matA = Array(typeBuffer)
    }

    public func copyBufToVal<T: Numeric>(_ bufA: MTLBuffer, _ a: inout T)
    {
        let size = MemoryLayout<T>.stride
        let typePtr = bufA.contents().bindMemory(to: T.self, capacity: size)
        let typeBuffer = UnsafeBufferPointer(start: typePtr, count: size)
        a = typeBuffer[0]
    }
}
