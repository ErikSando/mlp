#include <iostream>

#include "host/profiling/Task.hpp"

namespace mlp {
    namespace host {
        void Task::start() {
            m_start = clock::now();
        }

        void Task::end() {
            m_count++;

            m_end = clock::now();

            float duration = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(m_end - m_start).count();

            if (duration < m_min || m_count == 1) m_min = duration;
            if (duration > m_max) m_max = duration;

            m_duration += duration;

            m_average = m_duration / (float) m_count;
        }
    }
}