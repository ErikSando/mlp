#pragma once

#include "data/Dataset.hpp"
#include "data/Sample.hpp"
#include "mlp/MLP.hpp"

namespace mlp {
    void CommandLoop();

    // void Classify(MLP<>& model, const Sample& sample);
    // void Test(MLP& model, Dataset& dataset);
    // void Train(MLP& model, Dataset& dataset, const size_t n_epochs);
}