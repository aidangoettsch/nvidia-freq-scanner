#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ unsigned int getGlobalIdx() {
    unsigned int blockId = blockIdx.x;
    unsigned int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

extern "C" __global__ void benchmarkWrite(unsigned long long* buffer, long long* threadSize) {
    unsigned int threadId = getGlobalIdx();

    unsigned long long* benchmarkBuffer = buffer + (threadId * (*threadSize));

    for (long long i = 0; i < *threadSize; i += 8) {
        benchmarkBuffer[i] = i;
        benchmarkBuffer[i + 1] = i + 1;
        benchmarkBuffer[i + 2] = i + 2;
        benchmarkBuffer[i + 3] = i + 3;
        benchmarkBuffer[i + 4] = i + 4;
        benchmarkBuffer[i + 5] = i + 5;
        benchmarkBuffer[i + 6] = i + 6;
        benchmarkBuffer[i + 7] = i + 7;
    }
}

extern "C" __global__ void benchmarkCopy(long long** a, long long** b, const long long* thread_size) {
    unsigned int threadId = getGlobalIdx();
    long long* aBuf = a[threadId];
    long long* bBuf = b[threadId];

    memcpy(aBuf, bBuf, sizeof(long long) * *thread_size);
}

extern "C" __global__ void benchmarkRead(long long** pointers, const long long* thread_size, long long* errors) {
    unsigned int threadId = getGlobalIdx();
    long long* benchmarkBuffer = pointers[threadId];
    long long errCount = 0;

    for (long long i = 0; i < *thread_size; i++) {
         errCount += (long long) (benchmarkBuffer[i] != i);
    }

    *errors += errCount;
}

int main() {
    return 0;
}