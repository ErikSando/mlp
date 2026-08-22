#include "host/Context.hpp"

namespace mlp {
    namespace host {
        void Context::transfer(Matrix_t& dest, const float* src) const {
            std::memcpy(dest.data(), src, dest.size() * sizeof(float));
        }

        void Context::transfer(float* dest, const Matrix_t& src) const {
            std::memcpy(dest, src.data(), src.size() * sizeof(float));
        }

        void Context::transfer(Matrix_t& dest, const Matrix_t& src) const {
            std::memcpy(dest.data(), src.data(), dest.size() * sizeof(float));
        }
    }
}