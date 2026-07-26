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

        size_t m_index = 1 << (sizeof(size_t) - 1);
    };

    class CUDAProfiler {
        public:

        CUDAProfiler(const char* name = "Profiler tasks") : m_name(name) {}

        CUDATaskID startTask(const char* name);
        void endTask(CUDATaskID id);

        void print();

        void enable() { m_enabled = true; }
        void disable() { m_enabled = false; }

        private:

        const char* m_name;

        bool m_enabled = true;
        bool m_inTask = false;

        std::vector<std::unique_ptr<CUDATask>> m_tasks;
    };
}