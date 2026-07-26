#pragma once

#include <chrono>

namespace mlp {
    using clock = std::chrono::steady_clock;
    using timestamp_t = clock::time_point;

    class Task {
        public:

        Task(const char* name) : m_name(name) {}

        void start();
        void end();

        void print();

        const char* getName() { return m_name; }
        float getDuration() { return m_duration; };

        private:

        const char* m_name;

        timestamp_t m_start;
        timestamp_t m_end;
        float m_duration;
    };
}