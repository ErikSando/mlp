#pragma once

#include <vector>

#include "profiling/HostTask.hpp"

/*
    Note to self: before you start profiling, implement some way for the CUDA functions to log the start and end so it's measured in GPU execution time
    Because CUDA kernel launches don't block the CPU
*/

namespace mlp {
    class HostProfiler {
        public:

        HostProfiler(const char* name = "Profiler tasks") : m_name(name) {}

        void startTask(const char* name);
        void endTask();

        void print();

        private:

        const char* m_name;

        bool m_inTask = false;

        std::vector<HostTask> m_tasks;
    };
}