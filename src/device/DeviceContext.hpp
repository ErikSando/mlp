#pragma once

#include <cassert>
#include <iostream>

#include "profiling/CUDAProfiler.hpp"

namespace mlp {
    enum class Activation { // not sure where to put this, leaving it here for now
        NONE,
        SIGMOID, TANH, RELU, LEAKY_RELU, // hidden layer activation functions
        SOFTMAX // output activation functions
    };

    enum class Loss {
        MSE, CCE
    };
}