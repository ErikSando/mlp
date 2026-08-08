#pragma once

#include <chrono>

#include <cuda_runtime.h>

namespace mlp {
    namespace cuda {
        class Task {
            public:

            Task(const char* name);
            ~Task();

            void start();
            void end();

            void print();

            const char* getName() { return m_name; }
            float getDuration();

            private:

            const char* m_name;

            cudaEvent_t m_start;
            cudaEvent_t m_end;
            float m_duration = -1;
        };
    }
}