#pragma once

#include "cuda/Context.hpp"
#include "data/Dataset.hpp"
#include "data/Sample.hpp"
#include "host/Context.hpp"
#include "mlp/MLP.hpp"
#include "opencl/Context.hpp"

#if !defined(RANDOM_BS) // using this just so intellisense leaves me alone
#define MLP_CUDA
#endif

namespace mlp {
    #if defined(MLP_CUDA)
    using Context = cuda::Context;
    using Profiler = cuda::Profiler;
    using MLP_t = MLP<Context>;
    #elif defined(MLP_OPENCL)
    using Context = opencl::Context;
    using Profiler = opencl::Profiler;
    using MLP_t = MLP<Context>;
    #elif defined(MLP_HOST)
    using Context = host::Context;
    using Profiler = host::Profiler;
    using MLP_t = MLP<Context>;
    #else
    #error "No compute context has been defined"
    #endif

    void command_loop();
}