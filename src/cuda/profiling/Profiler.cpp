#include <format>
#include <iostream>

#include "cuda/profiling/Profiler.hpp"

namespace mlp {
    namespace cuda {
        constexpr size_t MAX_FLOAT_LENGTH = 8;
        constexpr size_t MIN_GAP = 2; // minimum number of spaces between the task name and duration info (e.g. 2 is name  | 0.5 ms)

        const std::string BENCHMARK_TASK_NAME = "Benchmark";

        void get_string(float value, std::string& str, size_t max_len) {
            for (int p = static_cast<int>(max_len); p >= 0; p--) {
                std::string f = std::format("{:.{}g}", value, p);

                if (f.size() <= max_len) {
                    str = f;
                    return;
                }
            }

            std::cerr << "Could not fit " << value << " within " << max_len << " characters.\n";
        }

        void print_task(const std::string& name, const Task& task, const int longest_name_length) {
            std::cout << name;

            for (int i = 0; i < MIN_GAP + longest_name_length - static_cast<int>(name.size()); i++) {
                std::cout << ' ';
            }

            std::string total, min, max, average;

            get_string(task.getDuration(), total, MAX_FLOAT_LENGTH);
            get_string(task.getMin(), min, MAX_FLOAT_LENGTH);
            get_string(task.getMax(), max, MAX_FLOAT_LENGTH);
            get_string(task.getAverage(), average, MAX_FLOAT_LENGTH);

            std::cout << "| " << total << " ms ";

            for (size_t i = 0; i < MAX_FLOAT_LENGTH + 1 - total.size(); i++) {
                std::cout << ' ';
            }

            if (name == BENCHMARK_TASK_NAME) {
                std::cout << "|\n";
                return;
            }

            std::cout << "| " << min << " ms ";

            for (size_t i = 0; i < MAX_FLOAT_LENGTH + 1 - min.size(); i++) {
                std::cout << ' ';
            }

            std::cout << "| " << max << " ms ";

            for (size_t i = 0; i < MAX_FLOAT_LENGTH + 1 - max.size(); i++) {
                std::cout << ' ';
            }

            std::cout << "| " << average << " ms ";

            for (size_t i = 0; i < MAX_FLOAT_LENGTH + 1 - average.size(); i++) {
                std::cout << ' ';
            }

            std::cout << "|\n";
        }

        void Profiler::startTask(const std::string& name) { // i think const char* is okay because im using literals, but maybe std::string is better in general
            if (name == BENCHMARK_TASK_NAME) {
                std::cout << "Tried to use reserved benchmark task name, using a different name\n";
                std::string new_name = "__";
                new_name += name;
                startTask(new_name);
                return;
            }

            if (!m_tasks.contains(name)) {
                m_taskOrder.push_back(name);
            }

            m_tasks[name].start(); // if there are no tasks with that name, one will be created
        }

        void Profiler::endTask(const std::string& name) {
            auto it = m_tasks.find(name);

            if (it == m_tasks.end()) {
                std::cerr << "Tried to end a task that hasn't been started: " << name << "\n";
                return;
            }

            it->second.end();
        }

        void Profiler::startBenchmark() {
            if (!m_tasks.contains(BENCHMARK_TASK_NAME)) {
                m_taskOrder.push_back(BENCHMARK_TASK_NAME);
            }

            m_tasks[BENCHMARK_TASK_NAME].start();
        }

        void Profiler::endBenchmark() {
            endTask(BENCHMARK_TASK_NAME);
        }

        void Profiler::clear() {
            m_tasks.clear();
            m_taskOrder.clear();
        }

        void Profiler::print() const {
            if (m_tasks.empty() || !m_enabled) return;

            int longest_name_length = 0;

            for (const auto& [name, task] : m_tasks) {
                longest_name_length = std::max(longest_name_length, static_cast<int>(name.size()));
            }

            std::cout << m_name << ":\n";

            std::vector<std::string> headings = { "Task name", "Total", "Min", "Max", "Average" };
            std::vector<int> heading_lengths;

            for (const std::string& heading : headings) {
                heading_lengths.push_back(static_cast<int>(heading.size()));
            }

            std::cout << headings[0];

            for (int i = 0; i < MIN_GAP + longest_name_length - heading_lengths[0]; i++) {
                std::cout << ' ';
            }

            for (size_t i = 1; i < headings.size(); i++) {
                std::cout << "| " << headings[i];

                for (int j = 0; j < MAX_FLOAT_LENGTH + 5 - static_cast<int>(headings[i].size()); j++) {
                    std::cout << ' ';
                }
            }

            std::cout << "|\n";

            const Task* benchmark_task = nullptr;

            for (const std::string& name : m_taskOrder) {
                auto it = m_tasks.find(name);

                if (it == m_tasks.end()) {
                    std::cerr << "Task order vector contains the name of a task that does not exist, skipping";
                    continue;
                }

                if (name == BENCHMARK_TASK_NAME) {
                    benchmark_task = &it->second;
                    continue;
                }

                print_task(name, it->second, longest_name_length);
            }

            if (benchmark_task) {
                print_task(BENCHMARK_TASK_NAME, *benchmark_task, longest_name_length);
            }
        }
    }
}