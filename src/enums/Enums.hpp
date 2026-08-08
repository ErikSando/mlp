#pragma once

#include <cassert>
#include <iostream>

namespace mlp {
    enum class Activation {
        NONE,
        SIGMOID, TANH, RELU, LEAKY_RELU, // hidden layer activation functions
        SOFTMAX // output activation functions
    };

    enum class Loss {
        MSE, CCE
    };
}