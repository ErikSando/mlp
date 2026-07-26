#include <cstring>
#include <iostream>

#include "profiling/CUDAProfiler.hpp"

namespace mlp {
    CUDATaskID CUDAProfiler::startTask(const char* name) {
        if (!m_enabled) return CUDATaskID();

        m_tasks.emplace_back(std::make_unique<CUDATask>(name));
        m_tasks.back()->start();

        return CUDATaskID(m_tasks.size() - 1);
    }

    void CUDAProfiler::endTask(CUDATaskID id) {
        if (!m_enabled) return;

        m_tasks[id.m_index]->end();
    }

    void CUDAProfiler::print() {
        std::cout << m_name << ":\n";

        float total = 0.0f;

        for (auto& task : m_tasks) {
            // const char* name = task.getName();
            // size_t name_len = std::strlen(name);
            float duration = task->getDuration();

            std::cout << task->getName() << ": " << duration << " ms\n";

            total += duration;
        }

        std::cout << "Total: " << total << " ms\n";
    }
}