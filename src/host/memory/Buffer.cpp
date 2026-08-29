#include <cstring>
#include <iostream>

#include "host/memory/Buffer.hpp"

namespace mlp {
    namespace host {
        Buffer::Buffer(const size_t size) : m_size(size) {
            m_data = malloc(m_size);

            if (m_data == nullptr) {
                throw std::runtime_error("Failed to allocate memory for host buffer.\n");
            }
        }

        Buffer::~Buffer() {
            if (m_data != nullptr) free(m_data);
        }

        void Buffer::zero() {
            std::memset(m_data, 0, m_size);
        }
    }
}