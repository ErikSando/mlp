#include <iostream>

#include "profiling/HostTask.hpp"

namespace mlp {
    void HostTask::start() {
        m_start = clock::now();
    }

    void HostTask::end() {
        m_end = clock::now();
        m_duration = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(m_end - m_start).count();
    }

    void HostTask::print() {
        std::cout << m_name << ": " << m_duration << " ms\n";
    }
}