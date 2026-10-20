#pragma once

#include "enums/Enums.hpp"

namespace mlp {
    class OptimiserConfig {
        public:

        OptimiserConfig(const OptimiserType optimiser_type = OptimiserType::BasicSGD) : type(optimiser_type) {}

        OptimiserType type;

        float learningRate = 0.01f;
    };
}