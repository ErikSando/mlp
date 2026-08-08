#include <cstring>
#include <iostream>

#include "profiling/HostProfiler.hpp"

namespace mlp {
    void HostProfiler::startTask(const char* name) {
        if (m_inTask) {
            std::cerr << "\033[31m" << "[Error]\033[0m Called startTask() with an unfinished task.";
            return;
        }

        m_tasks.emplace_back(name);
        m_tasks.back().start();
        m_inTask = true;
    }

    void HostProfiler::endTask() {
        if (!m_inTask) {
            std::cerr << "\033[31m" << "[Error]\033[0m Called endTask() without an active task.";
            return;
        }

        m_tasks.back().end();
        m_inTask = false;
    }

    void HostProfiler::print() {
        std::cout << m_name << ":\n";

        float total = 0.0f;

        for (HostTask& task : m_tasks) {
            // const char* name = task.getName();
            // size_t name_len = std::strlen(name);
            float duration = task.getDuration();

            std::cout << task.getName() << ": " << duration << " ms\n";

            total += duration;
        }

        std::cout << "Total: " << total << " ms\n";
    }
}