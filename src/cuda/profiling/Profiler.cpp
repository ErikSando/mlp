#include <cstring>
#include <iostream>

#include "cuda/profiling/Profiler.hpp"

namespace mlp {
    namespace cuda {
        TaskID Profiler::startTask(const char* name) {
            if (!m_enabled) return TaskID();

            m_tasks.emplace_back(std::make_unique<Task>(name));
            m_tasks.back()->start();

            return TaskID(m_tasks.size() - 1);
        }

        void Profiler::endTask(TaskID id) {
            if (!m_enabled) return;

            m_tasks[id.m_index]->end();
        }

        void Profiler::print() {
            std::cout << m_name << ":\n";

            float total = 0.0f;

            for (auto& task : m_tasks) {
                float duration = task->getDuration();

                std::cout << task->getName() << ": " << duration << " ms\n";

                total += duration;
            }

            std::cout << "Total: " << total << " ms\n";
        }
    }
}