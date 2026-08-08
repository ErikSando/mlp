#include <iostream>

#include "cuda/profiling/Task.hpp"

namespace mlp {
    namespace cuda {
        Task::Task(const char* name) : m_name(name) {
            cudaEventCreate(&m_start);
            cudaEventCreate(&m_end);
        }

        Task::~Task() {
            cudaEventDestroy(m_start);
            cudaEventDestroy(m_end);
        }

        void Task::start() {
            cudaEventRecord(m_start);
        }

        void Task::end() {
            cudaEventRecord(m_end);
        }

        float Task::getDuration() {
            if (m_duration > 0) return m_duration;

            cudaEventSynchronize(m_end);
            cudaEventElapsedTime(&m_duration, m_start, m_end);

            return m_duration;
        }

        void Task::print() {
            std::cout << m_name << ": " << m_duration << " ms\n";
        }
    }
}