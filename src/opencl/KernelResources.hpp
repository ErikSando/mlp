#pragma once

#include "CL/cl.h"

namespace mlp {
    namespace opencl {
        struct KernelResources {
            cl_device_id deviceID;
            cl_context clContext;
        };
    }
}