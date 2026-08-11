#pragma once

namespace mlp {
    namespace opencl {
        class Profiler {
            public:

            Profiler(const char* name = "Profiler tasks") : m_name(name) {}

            void startTask(const char* name);
            void endTask();

            void print();

            private:

            const char* m_name;
        };
    }
}