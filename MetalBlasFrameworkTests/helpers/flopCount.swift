//
//  flopCount.swift
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-08.
//

func getAxpyGflopCount(N: Int) -> Double
{
    // 1 multiplication, 1 addition. Convert to Gflop
    return 2.0 * Double(N) / 1e9
}

func getScalGflopCount(N: Int) -> Double
{
    return 1.0 * Double(N) / 1e9
}

func getAsumGflopCount(N: Int) -> Double
{
    return 1.0 * Double(N) / 1e9
}

func getGemvGflopCount(transA: TransposeType, M: Int, N: Int) -> Double
{
    // Including scalar multiplications
    let K: Int = (transA == .NoTranspose) ? M : N
    return (2.0 * Double(M) * Double(N) + 2.0 * Double(K)) / 1e9
}

func getGemmGflopCount(M: Int, N: Int, K: Int) -> Double
{
    // First term for matmul, second term for scaling/adding C matrix
    return (2.0 * Double(M) * Double(N) * Double(K) + (2.0 * Double(M) * Double(N))) / 1e9
}
