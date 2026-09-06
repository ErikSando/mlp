#pragma once

#include "profiling/Profiler.hpp"

namespace mlp {
    struct Profiler::Impl {
        cl_command_queue commandQueue;
    };
}