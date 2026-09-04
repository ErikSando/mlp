#pragma once

#include <cassert>
#include <iostream>
#include <map>

namespace mlp {
    enum class Activation {
        NONE,
        LEAKY_RELU, RELU, SIGMOID, TANH, // hidden layer activation functions
        SOFTMAX // output activation functions
    };

    enum class Loss {
        CCE, MSE
    };

    enum class OutputActivationLossPair { // long name but i want it to have a clear purpose
            NONE_CCE, NONE_MSE,
            SOFTMAX_CCE, SOFTMAX_MSE
    };

    using OALP = OutputActivationLossPair;

    // not sure what cpp file to define it in so im using inline

    // Output activation function + loss function pairs
    inline const std::map<std::pair<Activation, Loss>, OALP> AL_PAIRS = {
        {{ Activation::NONE, Loss::CCE }, OALP::NONE_CCE},
        {{ Activation::NONE, Loss::MSE }, OALP::NONE_MSE},
        {{ Activation::SOFTMAX, Loss::CCE }, OALP::SOFTMAX_CCE},
        {{ Activation::SOFTMAX, Loss::MSE }, OALP::SOFTMAX_MSE},
    };
}