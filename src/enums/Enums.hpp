#pragma once

#include <cassert>
#include <iostream>

namespace mlp {
    enum class Activation {
        NONE,
        LEAKY_RELU, RELU, SIGMOID, TANH, // hidden layer activation functions
        SOFTMAX // output activation functions
    };

    enum class Loss {
        CCE, MSE
    };
}