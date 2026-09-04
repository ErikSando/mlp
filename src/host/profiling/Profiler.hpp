#pragma once

#include <unordered_map>
#include <vector>

#include "host/profiling/Task.hpp"

namespace mlp {
    namespace host {
        class Profiler {
            public:

            Profiler(const std::string& name = "Profiler tasks") : m_name(name) {}

            void startTask(const std::string& name);
            void endTask(const std::string& name);

            void startBenchmark(); // measures the total GPU time spent between the start and end of the benchmarking period
            void endBenchmark();

            void reset();

            void print() const;

            void enable() { m_enabled = true; }
            void disable() { m_enabled = false; }

            bool enabled() const { return m_enabled; }

            private:

            std::string m_name;

            bool m_enabled = true;

            std::unordered_map<std::string, Task> m_tasks;

            std::vector<std::string> m_taskOrder;
        };
    }
}