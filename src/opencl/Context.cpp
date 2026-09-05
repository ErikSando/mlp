#include "opencl/Context.hpp"
#include "opencl/CLstuff.hpp"

namespace mlp {
    namespace opencl {
        Context::Context(Profiler* profiler) : m_profiler(profiler) {
            cl_int err;
            cl_uint num_platforms;

            err = clGetPlatformIDs(0, nullptr, &num_platforms);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to get platform IDs");
            }

            std::cout << num_platforms << " platform/s\n";

            err = clGetPlatformIDs(num_platforms, platform_ids, nullptr);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to get platform IDs");
            }

            char version[128];
            clGetPlatformInfo(platform_ids[0], CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
            std::cout << "OpenCL version: " << version << "\n";

            err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to get device IDs");
            }

            clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
            clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc_size), &max_alloc_size, nullptr);

            std::cout << "Global memory size: " << global_mem_size << "\n";
            std::cout << "Max mem alloc size: " << max_alloc_size << "\n";

            clcontext = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);

            if (!clcontext) {
                CL_ERROR(err, "Failed to create context");
                clReleaseContext(clcontext);
            }

            const cl_queue_properties queue_props[] = {
                CL_QUEUE_PROPERTIES,
                #ifndef NDEBUG
                CL_QUEUE_PROFILING_ENABLE,
                #endif
                0
            };

            command_queue = clCreateCommandQueueWithProperties(clcontext, device_id, queue_props, &err);

            if (!command_queue) {
                CL_ERROR(err, "Failed to create command queue");
            }
        }
    }
}