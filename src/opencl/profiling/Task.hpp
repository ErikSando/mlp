#pragma once

#include <vector>

#include "CL/cl.h"

namespace mlp {
    namespace opencl {
        struct EventPair {
            cl_event start;
            cl_event end;
        };

        class Task {
            public:

            Task(const cl_command_queue& command_queue);
            ~Task();

            void start();
            void end();

            float getDuration();
            float getMin();
            float getMax();
            float getAverage();

            private:

            void resolve();

            std::vector<EventPair> m_eventPairs;

            cl_command_queue m_commandQueue;
            cl_event m_lastStart;

            float m_duration = 0.0f;
            float m_min = 0.0f;
            float m_max = 0.0f;
            float m_average = 0.0f;

            size_t m_count = 0;
        };
    }
}