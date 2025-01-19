//
//  metalReduce.h
//  MetalBlas
//
//  Created by Daine McNiven on 2025-01-18.
//

#ifndef METAL_REDUCE_H
#define METAL_REDUCE_H

// TODO: do I need this as a header
template <typename T>
void metalsthreadgroupReduce(const device int& N [[buffer(0)]],
                             device T* w[[buffer(1)]],
                             device T* res[[buffer(2)]],
                             uint gid [[thread_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tgsize [[threads_per_threadgroup]]);

#endif

