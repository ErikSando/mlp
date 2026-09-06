#include <memory>
#include <iostream>

#include "cuda/profiling/ProfilerImpl.hpp"
#include "profiling/Profiler.hpp"

namespace mlp {
    Profiler::Profiler(const std::string& name) : m_name(name), m_impl(std::make_unique<Impl>()) {}

    Profiler::~Profiler() = default;

    void Profiler::newTask(const std::string& name) {
        m_tasks.try_emplace(name);
    }
}