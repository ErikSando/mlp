#include <algorithm>
#include <format>
#include <iostream>

#include "profiling/Profiler.hpp"

namespace mlp {
    inline void get_string(float value, std::string& str, int max_len) {
        for (int p = max_len; p >= 1; p--) {
            std::string f = std::format("{:.{}g}", value, p);

            if (f.size() <= static_cast<size_t>(max_len)) {
                str = f;
                return;
            }
        }

        std::cerr << "Could not fit " << value << " within " << max_len << " characters.\n";
    }

    void print_task(const std::string& name, Task& task, const int longest_name_length) {
        std::cout << name;

        for (int i = 0; i < profiler_settings::MIN_GAP + longest_name_length - static_cast<int>(name.size()); i++) {
            std::cout << ' ';
        }

        std::string total, min, max, average;

        get_string(task.getDuration(), total, profiler_settings::MAX_FLOAT_LENGTH);
        get_string(task.getMin(), min, profiler_settings::MAX_FLOAT_LENGTH);
        get_string(task.getMax(), max, profiler_settings::MAX_FLOAT_LENGTH);
        get_string(task.getAverage(), average, profiler_settings::MAX_FLOAT_LENGTH);

        std::cout << "| " << total << " ms ";

        for (int i = 0; i < profiler_settings::MAX_FLOAT_LENGTH + 1 - static_cast<int>(total.size()); i++) {
            std::cout << ' ';
        }

        if (name == profiler_settings::BENCHMARK_TASK_NAME) {
            std::cout << "|\n";
            return;
        }

        std::cout << "| " << min << " ms ";

        for (int i = 0; i < profiler_settings::MAX_FLOAT_LENGTH + 1 - static_cast<int>(min.size()); i++) {
            std::cout << ' ';
        }

        std::cout << "| " << max << " ms ";

        for (int i = 0; i < profiler_settings::MAX_FLOAT_LENGTH + 1 - static_cast<int>(max.size()); i++) {
            std::cout << ' ';
        }

        std::cout << "| " << average << " ms ";

        for (int i = 0; i < profiler_settings::MAX_FLOAT_LENGTH + 1 - static_cast<int>(average.size()); i++) {
            std::cout << ' ';
        }

        std::cout << "|\n";
    }

    void Profiler::startTask(const std::string& name) {
        if (!m_enabled) return;

        if (name == profiler_settings::BENCHMARK_TASK_NAME) {
            std::cout << "Tried to use reserved benchmark task name, using a different name\n";
            std::string new_name = "__" + name;
            startTask(new_name);
            return;
        }

        if (!m_tasks.contains(name)) {
            m_taskOrder.push_back(name);
            newTask(name);
        }

        m_tasks.at(name).start();
    }

    void Profiler::endTask(const std::string& name) {
        if (!m_enabled) return;

        auto it = m_tasks.find(name);

        if (it == m_tasks.end()) {
            std::cerr << "Tried to end a task that hasn't been started: " << name << "\n";
            return;
        }

        it->second.end();
    }

    void Profiler::startBenchmark() {
        if (!m_tasks.contains(profiler_settings::BENCHMARK_TASK_NAME)) {
            m_taskOrder.push_back(profiler_settings::BENCHMARK_TASK_NAME);
        }

        m_tasks.at(profiler_settings::BENCHMARK_TASK_NAME).start();
    }

    void Profiler::endBenchmark() {
        endTask(profiler_settings::BENCHMARK_TASK_NAME);
    }

    void Profiler::clear() {
        m_tasks.clear();
        m_taskOrder.clear();
    }

    void Profiler::print() {
        if (m_tasks.empty() || !m_enabled) {
            std::cout << m_name << ": no tasks to print\n";
            return;
        }

        int longest_name_length = 0;

        for (const auto& [name, task] : m_tasks) {
            longest_name_length = std::max(longest_name_length, static_cast<int>(name.size()));
        }

        std::cout << m_name << ":\n";

        std::vector<std::string> headings = { "Task name", "Total", "Min", "Max", "Average" };
        std::vector<int> heading_lengths;

        for (const std::string& heading : headings) {
            heading_lengths.push_back(static_cast<int>(heading.size()));
            longest_name_length = std::max(longest_name_length, static_cast<int>(heading.size()));
        }

        std::cout << headings[0];

        for (int i = 0; i < profiler_settings::MIN_GAP + longest_name_length - heading_lengths[0]; i++) {
            std::cout << ' ';
        }

        for (size_t i = 1; i < headings.size(); i++) {
            std::cout << "| " << headings[i];

            for (int j = 0; j < profiler_settings::MAX_FLOAT_LENGTH + 5 - heading_lengths[i]; j++) {
                std::cout << ' ';
            }
        }

        std::cout << "|\n";

        Task* benchmark_task = nullptr;

        for (const std::string& name : m_taskOrder) {
            auto it = m_tasks.find(name);

            if (it == m_tasks.end()) {
                std::cerr << "Task order vector contains the name of a task that does not exist, skipping";
                continue;
            }

            if (name == profiler_settings::BENCHMARK_TASK_NAME) {
                benchmark_task = &it->second;
                continue;
            }

            print_task(name, it->second, longest_name_length);
        }

        if (benchmark_task) {
            print_task(profiler_settings::BENCHMARK_TASK_NAME, *benchmark_task, longest_name_length);
        }
    }
}