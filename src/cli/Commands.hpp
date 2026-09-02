#pragma once

#include "cli/CLI.hpp"
#include "data/ParseSample.hpp"
#include "data/Sample.hpp"
#include "mlp/MLP.hpp"

namespace mlp {
    struct TestData {
        void reset() {
            incorrect = 0UL;
            correct = 0UL;
        }

        float getAccuracy() { return (float) correct / (float) (correct + incorrect); }

        size_t incorrect;
        size_t correct;
    };

    namespace commands {
        void train(MLP_t* model, Dataset& dataset, size_t n_epochs);
        void test(MLP_t* model, Context& context, Dataset& dataset, TestData& data);

        void print_sample(const ImageData& sample);
        void classify_sample(MLP_t* model, const ImageData& image_data, const int label = NO_LABEL);
    }
}