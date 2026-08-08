#pragma once

#include <cstddef>
#include <vector>

#include "matrix/Matrix.hpp"

namespace mlp {
    class HostMatrix : public IMatrix {
        public:

        HostMatrix(size_t rows, size_t columns) : IMatrix(rows, columns), m_data(rows, columns) {}
        ~HostMatrix();

        void zero() override;

        float* data() override { return m_data.data(); }
        const float* data() const override { return m_data.data(); }

        private:

        std::vector<float> m_data;
    };
}