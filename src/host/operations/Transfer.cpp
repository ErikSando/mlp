#include "host/Context.hpp"

namespace mlp {
    namespace host {
        void Context::transfer(Matrix_t& dest, const float* src) const {
            if (m_profiler) m_profiler->startTask("Upload (memcpy)");
            std::memcpy(dest.data(), src, dest.size() * sizeof(float));
            if (m_profiler) m_profiler->endTask("Upload (memcpy)");
        }

        void Context::transfer(float* dest, const Matrix_t& src) const {
            if (m_profiler) m_profiler->startTask("Download (memcpy)");
            std::memcpy(dest, src.data(), src.size() * sizeof(float));
            if (m_profiler) m_profiler->endTask("Download (memcpy)");
        }

        void Context::transfer(Matrix_t& dest, const Matrix_t& src) const {
            if (m_profiler) m_profiler->startTask("Copy (memcpy)");
            std::memcpy(dest.data(), src.data(), dest.size() * sizeof(float));
            if (m_profiler) m_profiler->endTask("Copy (memcpy)");
        }

        void Context::transfer(Buffer_t& dest, const void* src) const {
            if (m_profiler) m_profiler->startTask("Upload (memcpy)");
            std::memcpy(dest.data(), src, dest.size());
            if (m_profiler) m_profiler->endTask("Upload (memcpy)");
        }

        void Context::transfer(void* dest, const Buffer_t& src) const {
            if (m_profiler) m_profiler->startTask("Download (memcpy)");
            std::memcpy(dest, src.data(), src.size());
            if (m_profiler) m_profiler->endTask("Download (memcpy)");
        }

        void Context::transfer(Buffer_t& dest, const Buffer_t& src) const {
            if (m_profiler) m_profiler->startTask("Copy (memcpy)");
            std::memcpy(dest.data(), src.data(), dest.size());
            if (m_profiler) m_profiler->endTask("Copy (memcpy)");
        }
    }
}