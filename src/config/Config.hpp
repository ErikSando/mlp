#pragma once

// #include "profiling/Profiler.hpp"

#if !defined(RANDOM_BS) // using this just so intellisense leaves me alone
#define MLP_HOST
#endif

#if defined(MLP_CUDA)

#include "cuda/Context.hpp"
#include "cuda/profiling/Task.hpp"

namespace mlp {
    // using Profiler = ProfilerT<cuda::Task>;
    // using Profiler = cuda::Profiler;
    using Context = cuda::Context;
    using Matrix = cuda::Matrix;
    using Buffer = cuda::Buffer;

    const inline std::string PROFILER_NAME = "CUDA profiler tasks";
}

#elif defined(MLP_OPENCL)

#include "opencl/Context.hpp"
#include "opencl/profiling/Task.hpp"

namespace mlp {
    // using Profiler = ProfilerT<opencl::Task>;
    // using Profiler = opencl::Profiler;
    using Context = opencl::Context;
    using Matrix = opencl::Matrix;
    using Buffer = opencl::Buffer;

    const inline std::string PROFILER_NAME = "OpenCL profiler tasks";
}

#elif defined(MLP_HOST)

#include "host/Context.hpp"
#include "host/profiling/Task.hpp"

namespace mlp {
    // using Profiler = ProfilerT<host::Task>;
    // using Profiler = host::Profiler;
    using Context = host::Context;
    using Matrix = host::Matrix;
    using Buffer = host::Buffer;

    const inline std::string PROFILER_NAME = "Host profiler tasks";
}

#else
#error "No compute context has been defined"
#endif