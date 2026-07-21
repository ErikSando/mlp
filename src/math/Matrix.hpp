#pragma once

#include <cstddef>

namespace mlp {
    template<size_t TRows, size_t TColumns>
    class Matrix {
        constexpr size_t size() { return TRows * TColumns; }
        constexpr size_t rows() { return TRows; }
        constexpr size_t columns() { return TColumns; }
    };
}