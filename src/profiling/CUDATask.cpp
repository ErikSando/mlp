#include <iostream>

#include "profiling/CUDATask.hpp"

namespace mlp {
    CUDATask::CUDATask(const char* name) : m_name(name) {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
    }

    CUDATask::~CUDATask() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_end);
    }

    void CUDATask::start() {
        cudaEventRecord(m_start);
    }

    void CUDATask::end() {
        cudaEventRecord(m_end);
    }

    float CUDATask::getDuration() {
        if (m_duration > 0) return m_duration;

        cudaEventSynchronize(m_end);
        cudaEventElapsedTime(&m_duration, m_start, m_end);
        return m_duration;
    }

    void CUDATask::print() {
        std::cout << m_name << ": " << m_duration << " ms\n";
    }
}