#pragma once

#include <memory>
#include <vector>

#include "cuda/profiling/Task.hpp"

namespace mlp {
    namespace cuda {
        struct TaskID {
            friend class Profiler;

            TaskID() = default;
            TaskID(size_t index) : m_index(index) {}

            private:

            size_t m_index = 1 << (sizeof(size_t) - 1);
        };

        class Profiler {
            public:

            Profiler(const char* name = "Profiler tasks") : m_name(name) {}

            TaskID startTask(const char* name);
            void endTask(TaskID id);

            void print();

            void enable() { m_enabled = true; }
            void disable() { m_enabled = false; }

            private:

            const char* m_name;

            bool m_enabled = true;
            bool m_inTask = false;

            std::vector<std::unique_ptr<Task>> m_tasks;
        };
    }
}