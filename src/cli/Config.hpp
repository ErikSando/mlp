#pragma once

#include <string>

#if !defined(RANDOM_BS) // using this just so intellisense leaves me alone
#define MLP_HOST
#endif

namespace mlp {
    #if defined(MLP_CUDA)

    const inline std::string PROFILER_NAME = "CUDA profiler tasks";

    #elif defined(MLP_OPENCL)

    const inline std::string PROFILER_NAME = "OpenCL profiler tasks";

    #elif defined(MLP_HOST)

    const inline std::string PROFILER_NAME = "Host profiler tasks";

    #else
    #error "No compute context has been defined"
    #endif
}