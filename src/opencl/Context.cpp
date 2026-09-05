#include "opencl/Context.hpp"

namespace mlp {
    namespace opencl {
        void Context::init() {
            cl_int err;
            cl_uint num_platforms;

            err = clGetPlatformIDs(0, nullptr, &num_platforms);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to get platform IDs");
            }

            std::cout << num_platforms << " platform/s\n";

            err = clGetPlatformIDs(num_platforms, m_platformIDs, nullptr);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to get platform IDs");
            }

            char version[128];
            clGetPlatformInfo(m_platformIDs[0], CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
            std::cout << "OpenCL version: " << version << "\n";

            err = clGetDeviceIDs(m_platformIDs[0], CL_DEVICE_TYPE_GPU, 1, &m_deviceID, nullptr);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to get device IDs");
            }

            clGetDeviceInfo(m_deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(m_globalMemSize), &m_globalMemSize, nullptr);
            clGetDeviceInfo(m_deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(m_maxAllocSize), &m_maxAllocSize, nullptr);

            std::cout << "Global memory size: " << m_globalMemSize << "\n";
            std::cout << "Max mem alloc size: " << m_maxAllocSize << "\n";

            m_clContext = clCreateContext(nullptr, 1, &m_deviceID, nullptr, nullptr, &err);

            if (!m_clContext) {
                CL_ERROR(err, "Failed to create context");
                clReleaseContext(m_clContext);
            }

            const cl_queue_properties queue_props[] = {
                CL_QUEUE_PROPERTIES,
                #ifndef NDEBUG
                CL_QUEUE_PROFILING_ENABLE,
                #endif
                0
            };

            m_commandQueue = clCreateCommandQueueWithProperties(m_clContext, m_deviceID, queue_props, &err);

            if (!m_commandQueue) {
                CL_ERROR(err, "Failed to create command queue");
            }
        }
    }
}