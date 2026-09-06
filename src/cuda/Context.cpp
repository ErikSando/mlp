#include "context/Context.hpp"
#include "cuda/ContextImpl.hpp"

namespace mlp {
    Context::Context(Profiler* profiler) : m_profiler(profiler), m_impl(std::make_unique<Impl>()) {}
    Context::~Context() = default;

    Matrix Context::createMatrix(const size_t rows, const size_t columns) const {
        return Matrix(rows, columns);
    }

    Buffer Context::createBuffer(const size_t size) const {
        return Buffer(size);
    }

    void Context::zeroMatrix(Matrix& matrix) const {
        matrix.zero();
    }

    void Context::zeroBuffer(Buffer& buffer) const {
        buffer.zero();
    }
}