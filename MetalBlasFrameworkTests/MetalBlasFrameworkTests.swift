////
////  MetalBlasFrameworkTests.swift
////  MetalBlasFrameworkTests
////
////  Created by Daine McNiven on 2025-01-04.
////
//
//import Testing
//@testable import MetalBlasFramework
//
//
//struct MetalBlasFrameworkTests {
//
//    @Test func example() async throws
//    {
//        let metalBlas = MetalBlas()
//        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
//        let storageMethod = StorageMethod.ColMajor
//        let transA = TransposeType.NoTranspose
//        let transB = transA
//        let M = 129
//        let N = 129
//        let K = 129
//        let lda = 129
//        let ldb = 129
//        let ldc = 129
//        let alpha : Float = 1
//        let beta : Float = 1
//        var matA : [Float] = []
//        var matB : [Float] = []
//        var matC : [Float] = []
//        let size = 129 * 129
//        
//        for _ in 0...size
//        {
//            matA.append(1)
//            matB.append(2)
//            matC.append(3)
//        }
//        let bufferA = metalBlas?.getDeviceBuffer(matA: &matA, M: size)
//        let bufferB = metalBlas?.getDeviceBuffer(matA: &matB, M: size)
//        var bufferC = metalBlas?.getDeviceBuffer(matA: &matC, M: size)
//        
//        metalBlas!.metalSgemm(storageMethod: storageMethod, transA: transA, transB: transB, M: M, N: N, K: K,
//                              alpha: alpha, matA: bufferA!, lda: lda, matB: bufferB!, ldb: ldb, beta: beta, matC: &bufferC!, ldc: ldc)
//        let sgemmOutputs = unsafeBitCast(bufferC?.contents(), to: UnsafeMutablePointer<Float>.self)
//        print("output;")
//        for i in 0...M - 1
//        {
//            for j in 0...N - 1
//            {
//                print(sgemmOutputs[j * ldc + i], terminator: " ")
//            }
//            print("")
//        }
//    }
//
//}
