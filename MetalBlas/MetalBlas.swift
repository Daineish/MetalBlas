//
//  MetalBlas.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-04.
//

import Metal

public enum TransposeType
{
    case NoTranspose
    case Transpose
    case ConjTranspose
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
                "hasum": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHasum")!),
                "sasum": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSasum")!),
                "haxpy": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHaxpy")!),
                "saxpy": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSaxpy")!),
                "hcopy": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHcopy")!),
                "scopy": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalScopy")!),
                "hdot": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHdot")!),
                "sdot": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSdot")!),
//                "ihamin": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalIhamin")!),
//                "isamin": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalIsamin")!),
                "ihamax": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalIhamax")!),
                "isamax": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalIsamax")!),
                "hnrm2": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHnrm2")!),
                "snrm2": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSnrm2")!),
                "hscal": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHscal")!),
                "sscal": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSscal")!),
                "hswap": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHswap")!),
                "sswap": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSswap")!),
                "hgemm": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalHgemm")!),
                "sgemm": try device.makeComputePipelineState(function: self.library.makeFunction(name: "metalSgemm")!),
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

    public func copyBufToArray<T: Numeric>(_ bufA: MTLBuffer, _ matA: inout [T])
    {
        // Copy the data into the array
        let size = bufA.length / MemoryLayout<T>.stride
        let bufferPointer = bufA.contents()
        let typePtr = bufferPointer.bindMemory(to: T.self, capacity: size)
        let typeBuffer = UnsafeBufferPointer(start: typePtr, count: size)
        matA = Array(typeBuffer)
    }
}

