#pragma once

#include "CL/cl.h"

#define ERROR(message)\
    do {\
        std::cout << "\033[31m" << "[Error]\033[0m "\
                  << message << '\n'\
                  << "File: " << __FILE__ << '\n'\
                  << "Line: " << __LINE__ << '\n'\
                  << "Function: " << __func__ << '\n';\
        std::abort();\
    } while (0);

#define CL_ERROR(err, message)\
    do {\
        std::cout << "\033[31m" << "[Error]\033[0m "\
                  << message << ": "\
                  << err << '\n'\
                  << "File: " << __FILE__ << '\n'\
                  << "Line: " << __LINE__ << '\n'\
                  << "Function: " << __func__ << '\n';\
        std::abort();\
    } while (0);

namespace mlp {
    namespace opencl {
        extern cl_platform_info platform_info;
        extern cl_platform_id platform_ids[100];
        extern cl_device_id device_id;
        extern cl_ulong global_mem_size;
        extern cl_ulong max_alloc_size;
        extern cl_context clcontext;
        extern cl_command_queue command_queue;
    }
}