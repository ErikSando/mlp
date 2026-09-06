#pragma once

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "profiling/TaskConfig.hpp"

namespace mlp {
    namespace profiler_settings {
        constexpr int MAX_FLOAT_LENGTH = 8;
        constexpr int MIN_GAP = 2; // minimum number of spaces between the task name and duration info (e.g. 2 is name  | 0.5 ms)

        inline const std::string BENCHMARK_TASK_NAME = "Benchmark";
    }

    class Context;

    class Profiler {
        friend class Context;

        public:

        Profiler(const std::string& name = "Profiler tasks");
        ~Profiler();

        void startTask(const std::string& name);
        void endTask(const std::string& name);

        void startBenchmark(); // measures the total time spent on the CUDA stream between the start and end of the benchmarking period
        void endBenchmark();

        void clear();

        void print();

        void enable() { m_enabled = true; }
        void disable() { m_enabled = false; }

        bool enabled() const { return m_enabled; }

        private:

        void newTask(const std::string& name);

        struct Impl;
        std::unique_ptr<Impl> m_impl;

        std::string m_name;

        bool m_enabled = true;

        std::unordered_map<std::string, Task> m_tasks;

        std::vector<std::string> m_taskOrder;
    };
}