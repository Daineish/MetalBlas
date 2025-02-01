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

func getAmaxGflopCount(N: Int) -> Double
{
    return 1.0 * Double(N) / 1e9
}

func getAminGflopCount(N: Int) -> Double
{
    return getAmaxGflopCount(N: N)
}

func getAsumGflopCount(N: Int) -> Double
{
    return 1.0 * Double(N) / 1e9
}

func getCopyGbyteCount(N: Int) -> Double
{
    return 1.0 * Double(N) / 1e9
}

func getDotGflopCount(N: Int) -> Double
{
    return 2.0 * Double(N) / 1e9
}

func getNrm2GflopCount(N: Int) -> Double
{
    return 2.0 * Double(N) / 1e9
}

func getRotGflopCount(N: Int) -> Double
{
    return 6.0 * Double(N) / 1e9
}

func getRotmGflopCount(N: Int, flag: Float) -> Double
{
    if flag == -2.0
    {
        return 0
    }
    else if flag < 0
    {
        return 6.0 * Double(N) / 1e9
    }
    else
    {
        return 4.0 * Double(N) / 1e9
    }
}

func getRotgGflopCount() -> Double
{
    return 10.0 // ish, kinda dumb
}

func getSwapGbyteCount(N: Int) -> Double
{
    return 2.0 * Double(N) / 1e9
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
