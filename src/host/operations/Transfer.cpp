#include "host/Context.hpp"

namespace mlp {
    namespace host {
        void Context::transfer(const float* src, Matrix_t& dest) const {
            std::memcpy(dest.data(), src, dest.size() * sizeof(float));
        }

        void Context::transfer(const Matrix_t& src, float* dest) const {
            std::memcpy(dest, src.data(), src.size() * sizeof(float));
        }

        void Context::transfer(const Matrix_t& src, Matrix_t& dest) const {
            std::memcpy(dest.data(), src.data(), dest.size() * sizeof(float));
        }
    }
}