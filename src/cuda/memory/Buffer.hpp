#pragma once

#include <cstddef>

namespace mlp {
    namespace cuda {
        class Buffer {
            public:

            Buffer(size_t size);
            ~Buffer();

            void zero();

            void* data() { return m_data; }
            const void* data() const { return m_data; }

            constexpr size_t size() const { return m_size; }

            private:

            size_t m_size;

            void* m_data = nullptr;
        };
    }
}