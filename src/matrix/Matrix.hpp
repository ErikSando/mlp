#pragma once

#include <cstddef>

namespace mlp {
    class Matrix {
        public:

        Matrix(const size_t rows, const size_t columns);
        ~Matrix();

        Matrix(const Matrix&) = delete;
        Matrix& operator=(const Matrix&) = delete;

        Matrix(Matrix&& other) noexcept;
        Matrix& operator=(Matrix&& other) noexcept;

        void resize(const size_t rows, const size_t columns);

        float* data() const { return m_data; } // device memory address of the matrix values

        constexpr size_t size() const { return m_size; }
        constexpr size_t rows() const { return m_rows; }
        constexpr size_t columns() const { return m_cols; }

        private:

        size_t m_size;
        size_t m_rows;
        size_t m_cols;

        float* m_data = nullptr;
    };

    struct DeviceInt {
        DeviceInt(const int value = 0);
        ~DeviceInt();

        int* data() { return m_data; }

        private:

        int* m_data = nullptr;
    };

    struct DeviceFloat {
        DeviceFloat(const float value = 0.0f);
        ~DeviceFloat();

        float* data() { return m_data; }

        private:

        float* m_data = nullptr;
    };
}