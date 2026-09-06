#include <memory>

#include "context/Context.hpp"
#include "opencl/ContextImpl.hpp"
#include "opencl/Error.hpp"
#include "opencl/ContextImpl.hpp"
#include "opencl/Kernel.hpp"
#include "opencl/profiling/ProfilerImpl.hpp"

namespace mlp {
    Context::Context(Profiler* profiler) : m_profiler(profiler), m_impl(std::make_unique<Impl>()) {
        cl_int err;
        cl_uint num_platforms;

        err = clGetPlatformIDs(0, nullptr, &num_platforms);

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to get platform IDs");
        }

        std::cout << num_platforms << " platform/s\n";

        err = clGetPlatformIDs(num_platforms, m_impl->platformIDs, nullptr);

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to get platform IDs");
        }

        char version[128];
        clGetPlatformInfo(m_impl->platformIDs[0], CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
        std::cout << "OpenCL version: " << version << "\n";

        err = clGetDeviceIDs(m_impl->platformIDs[0], CL_DEVICE_TYPE_GPU, 1, &m_impl->deviceID, nullptr);

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to get device IDs");
        }

        clGetDeviceInfo(m_impl->deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(m_impl->globalMemSize), &m_impl->globalMemSize, nullptr);
        clGetDeviceInfo(m_impl->deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(m_impl->maxAllocSize), &m_impl->maxAllocSize, nullptr);

        std::cout << "Global memory size: " << m_impl->globalMemSize << "\n";
        std::cout << "Max mem alloc size: " << m_impl->maxAllocSize << "\n";

        m_impl->clContext = clCreateContext(nullptr, 1, &m_impl->deviceID, nullptr, nullptr, &err);

        if (!m_impl->clContext) {
            CL_ERROR(err, "Failed to create CL context");
            clReleaseContext(m_impl->clContext);
        }

        const cl_queue_properties queue_props[] = {
            CL_QUEUE_PROPERTIES,
            CL_QUEUE_PROFILING_ENABLE,
            0
        };

        m_impl->commandQueue = clCreateCommandQueueWithProperties(m_impl->clContext, m_impl->deviceID, queue_props, &err);

        if (!m_impl->commandQueue) {
            CL_ERROR(err, "Failed to create command queue");
        }

        opencl::KernelResources kernelResources{ m_impl->deviceID, m_impl->clContext };

        // m_impl->kernels.push_back(std::make_unique<opencl::Kernel>(kernelResources, opencl::kernel_paths::PROPAGATION, "propagation"));

        m_impl->kernels[opencl::KernelID::PROPAGATION] = std::make_unique<opencl::Kernel>(kernelResources, opencl::kernel_paths::PROPAGATION, "propagate");

        if (m_profiler) {
            m_profiler->m_impl->commandQueue = m_impl->commandQueue;
        }
    }

    Context::~Context() = default;

    Matrix Context::createMatrix(const size_t rows, const size_t columns) const {
        return Matrix(m_impl->clContext, rows, columns);
    }

    Buffer Context::createBuffer(const size_t size) const {
        return Buffer(m_impl->clContext, size);
    }

    void Context::zeroMatrix(Matrix& matrix) const {
        matrix.zero(m_impl->commandQueue);
    }

    void Context::zeroBuffer(Buffer& buffer) const {
        buffer.zero(m_impl->commandQueue);
    }
}