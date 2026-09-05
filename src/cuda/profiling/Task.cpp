#include <iostream>

#include "cuda/profiling/Task.hpp"

namespace mlp {
    namespace cuda {
        Task::Task() {
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

            m_count++;

            float duration;

            cudaEventSynchronize(m_end);
            cudaEventElapsedTime(&duration, m_start, m_end);

            if (duration < m_min || m_count == 1) m_min = duration;
            if (duration > m_max) m_max = duration;

            m_duration += duration;

            m_average = m_duration / (float) m_count;
        }
    }
}