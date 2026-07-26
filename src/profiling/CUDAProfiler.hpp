#pragma once

#include <memory>
#include <vector>

#include "profiling/CUDATask.hpp"

namespace mlp {
    struct CUDATaskID {
        friend class CUDAProfiler;

        CUDATaskID() = default;
        CUDATaskID(size_t index) : m_index(index) {}

        private:

        size_t m_index;
    };

    class CUDAProfiler {
        public:

        CUDAProfiler(const char* name = "Profiler tasks") : m_name(name) {}

        CUDATaskID startTask(const char* name);
        void endTask(CUDATaskID id);

        void print();

        private:

        const char* m_name;

        bool m_inTask = false;

        std::vector<std::unique_ptr<CUDATask>> m_tasks;
    };
}