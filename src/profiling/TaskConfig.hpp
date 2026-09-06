#pragma once

#if !defined(RANDOM_BS) // using this just so intellisense leaves me alone
#define MLP_OPENCL
#endif

#if defined(MLP_CUDA)

#include "cuda/profiling/Task.hpp"

namespace mlp {
    using Task = cuda::Task;
}

#elif defined(MLP_OPENCL)

#include "opencl/profiling/Task.hpp"

namespace mlp {
    using Task = opencl::Task;
}

#elif defined(MLP_HOST)

#include "host/profiling/Task.hpp"

namespace mlp {
    using Task = host::Task;
}

#else
#error "No compute context has been defined"
#endif