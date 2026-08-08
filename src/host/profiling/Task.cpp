#include <iostream>

#include "host/profiling/Task.hpp"

namespace mlp {
    namespace host {
        void Task::start() {
            m_start = clock::now();
        }

        void Task::end() {
            m_end = clock::now();
            m_duration = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(m_end - m_start).count();
        }

        void Task::print() {
            std::cout << m_name << ": " << m_duration << " ms\n";
        }
    }
}