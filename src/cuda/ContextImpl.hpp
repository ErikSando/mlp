#pragma once

#include <iostream>

#include "context/Context.hpp"

#define CUDA_ERROR(err, message)\
    do {\
        std::cout << "\033[31m" << "[Error]\033[0m "\
                  << message << ": "\
                  << cudaGetErrorString(err) << '\n'\
                  << "File: " << __FILE__ << '\n'\
                  << "Line: " << __LINE__ << '\n'\
                  << "Function: " << __func__ << '\n';\
        std::abort();\
    } while (0);

namespace mlp {
    constexpr unsigned int TILE_SIZE = 32;
    constexpr unsigned int BLOCK_SIZE = 256;

    inline unsigned int block_count(unsigned int thread_count, unsigned int block_size) {
        return (thread_count + block_size - 1) / block_size;
    }

    struct Context::Impl {};
}