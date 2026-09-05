#include "opencl/CLstuff.hpp"

namespace mlp {
    namespace opencl {
        cl_platform_info platform_info;
        cl_platform_id platform_ids[100];
        cl_device_id device_id;
        cl_ulong global_mem_size;
        cl_ulong max_alloc_size;
        cl_context clcontext;
        cl_command_queue command_queue;
    }
}