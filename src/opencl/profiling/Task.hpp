#pragma once

namespace mlp {
    namespace opencl {
        class Task {
            public:

            Task();
            ~Task();

            void start();
            void end();

            float getDuration() const { return m_duration; };
            float getMin() const { return m_min; }
            float getMax() const { return m_max; }
            float getAverage() const { return m_average; }

            private:

            // cudaEvent_t m_start;
            // cudaEvent_t m_end;

            float m_duration = 0.0f;
            float m_min = 0.0f;
            float m_max = 0.0f;
            float m_average = 0.0f;

            size_t m_count = 0;
        };
    }
}