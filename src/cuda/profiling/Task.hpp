#pragma once

#include <chrono>

#include <cuda_runtime.h>

namespace mlp {
    namespace cuda {
        // class Task {
        //     public:

        //     Task(const char* name);
        //     ~Task();

        //     void start();
        //     void end();

        //     void print();

        //     const char* getName() { return m_name; }
        //     float getDuration();

        //     private:

        //     const char* m_name;

        //     cudaEvent_t m_start;
        //     cudaEvent_t m_end;
        //     float m_duration = -1;
        // };

        class Task {
            public:

            // Task(const std::string name); // probably dont need the name, its tracked in the profiler's std::unordered_map
            Task();
            ~Task();

            void start();
            void end();

            // void print();

            // const std::string& getName() const { return m_name; }
            float getDuration() const { return m_duration; };
            float getMin() const { return m_min; }
            float getMax() const { return m_max; }
            float getAverage() const { return m_average; }

            private:

            // std::string m_name;

            cudaEvent_t m_start;
            cudaEvent_t m_end;

            float m_duration = 0.0f;
            float m_min = 0.0f;
            float m_max = 0.0f;
            float m_average = 0.0f;

            size_t m_count = 0;
        };
    }
}