#pragma once

#if defined(MLP_CUDA)

#include "cuda/memory/Buffer.hpp"
#include "cuda/memory/Matrix.hpp"

namespace mlp {
    using Buffer = cuda::Buffer;
    using Matrix = cuda::Matrix;
}

#elif defined(MLP_OPENCL)

#include "opencl/memory/Buffer.hpp"
#include "opencl/memory/Matrix.hpp"

namespace mlp {
    using Buffer = opencl::Buffer;
    using Matrix = opencl::Matrix;
}

#elif defined(MLP_HOST)

#include "host/memory/Buffer.hpp"
#include "host/memory/Matrix.hpp"

namespace mlp {
    using Buffer = host::Buffer;
    using Matrix = host::Matrix;
}

#else
#error "No compute context has been defined"
#endif