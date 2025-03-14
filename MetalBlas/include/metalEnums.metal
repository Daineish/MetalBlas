//
//  metalEnums.metal
//  MetalBlas
//
//  Created by Daine McNiven on 2025-02-01.
//

#ifndef Metalenums_h
#define Metalenums_h

typedef enum OrderType : uint32_t
{
    ColMajor  = 0,
    RowMajor  = 1,
} OrderType;

typedef enum TransposeType : uint32_t
{
    NoTranspose = 0,
    Transpose  = 1,
    ConjTranspose = 2
} TransposeType;

typedef enum UploType : uint32_t
{
    FillUpper,
    FillLower,
} UploType;

typedef enum DiagType : uint32_t
{
    NonUnit,
    Unit
} DiagType;

#endif
