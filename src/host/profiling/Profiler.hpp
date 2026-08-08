#pragma once

#include <vector>

#include "host/profiling/Task.hpp"

namespace mlp {
    namespace host {
        class Profiler {
            public:

            Profiler(const char* name = "Profiler tasks") : m_name(name) {}

            void startTask(const char* name);
            void endTask();

            void print();

            private:

            const char* m_name;

            bool m_inTask = false;

            std::vector<Task> m_tasks;
        };
    }
}