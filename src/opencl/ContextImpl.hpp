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
        }

        struct KernelID {
            enum : size_t {
                PROPAGATION,
                BACKPROP_OUTPUTS,
                BACKPROP_HIDDENS,
                OPTIMISE,
                // CHECK_OUTPUTS,
                COUNT
            };
        };
    }

    struct Context::Impl {
        cl_platform_info platformInfo;
        cl_platform_id platformIDs[100];
        cl_device_id deviceID;
        cl_ulong globalMemSize;
        cl_ulong maxAllocSize;
        cl_context clContext;
        cl_command_queue commandQueue;

        std::vector<std::unique_ptr<opencl::Kernel>> kernels;

        Impl() : kernels(opencl::KernelID::COUNT) {}
    };
}