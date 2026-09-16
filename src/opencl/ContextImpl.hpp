#pragma once

#include <memory>

#include "CL/cl.h"

#include "context/Context.hpp"
#include "opencl/Error.hpp"
#include "opencl/Kernel.hpp"

namespace mlp {
    namespace opencl {
        namespace kernel_paths {
            inline const std::string PROPAGATION = std::string(MLP_CL_KERNEL_DIR) + "Propagation.cl";
            inline const std::string SOFTMAX = std::string(MLP_CL_KERNEL_DIR) + "Softmax.cl";
            inline const std::string CHECK_OUTPUTS = std::string(MLP_CL_KERNEL_DIR) + "CheckOutputs.cl";
        }

        struct KernelID {
            enum : size_t {
                PROPAGATION,
                SOFTMAX,
                BACKPROP_OUTPUTS,
                BACKPROP_HIDDENS,
                OPTIMISE,
                CHECK_OUTPUTS,
                COUNT
            };
        };
    }

    constexpr size_t LOCAL_WORK_SIZE = 16;

    inline size_t global_work_size(size_t local_work_size, size_t dimension) {
        return (dimension + local_work_size - 1) / local_work_size * local_work_size; // division of integers rounds down
    }

    using Kernel_up = std::unique_ptr<opencl::Kernel>;

    struct Context::Impl {
        cl_platform_info platformInfo;
        cl_platform_id platformIDs[100];
        cl_device_id deviceID;
        cl_ulong globalMemSize;
        cl_ulong maxAllocSize;
        cl_context clContext;
        cl_command_queue commandQueue;

        std::vector<Kernel_up> kernels;

        Impl() : kernels(opencl::KernelID::COUNT) {}

        template<typename T>
        void setKernelArg(const Kernel_up& kernel, const cl_uint index, const T& arg) {
            cl_int err = kernel->setArgument(index, arg);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to set kernel argument " + index);
            }
        }

        template<typename T>
        void setKernelArg(const Kernel_up& kernel, const cl_uint index, const T& arg, const std::string& err_msg) {
            cl_int err = kernel->setArgument(index, arg);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, err_msg);
            }
        }
    };
}