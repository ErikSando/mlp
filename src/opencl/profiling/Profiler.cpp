#include <memory>

#include "profiling/Profiler.hpp"
#include "opencl/profiling/ProfilerImpl.hpp"

namespace mlp {
    Profiler::Profiler(const std::string& name) : m_name(name), m_impl(std::make_unique<Impl>()) {}

    Profiler::~Profiler() = default;

    void Profiler::newTask(const std::string& name) {
        m_tasks.try_emplace(name, m_impl->commandQueue);
    }
}