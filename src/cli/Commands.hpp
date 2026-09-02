#pragma once

#include <unordered_map>

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

    std::string activation_string(const Activation activation);
    std::string loss_string(const Loss loss);

    extern std::unordered_map<std::string, Activation> activation_functions; // maybe seperate hidden and output activation functions
    extern std::unordered_map<std::string, Loss> loss_functions;

    namespace commands {
        void train(MLP_t* model, Dataset& dataset, size_t n_epochs);
        void test(MLP_t* model, Dataset& dataset, TestData& data);

        void print_sample(const ImageData& sample);
        void classify_sample(MLP_t* model, const ImageData& image_data, const int label = NO_LABEL);

        constexpr char seperator = ' ';

        // bool save(MLP_t* model, std::string save_path);
        // bool load(MLP_t* model, std::string save_path);
        bool save(const ModelData& data, std::string save_path);
        bool load(ModelData& data, std::string save_path);
    }
}