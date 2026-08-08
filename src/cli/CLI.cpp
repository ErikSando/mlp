#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli/CLI.hpp"
#include "cuda/Context.hpp"
#include "cuda/profiling/Profiler.hpp"
#include "data/Dataset.hpp"
#include "host/Context.hpp"
#include "host/profiling/Profiler.hpp"
#include "mlp/MLP.hpp"

namespace mlp {
    std::string to_lower(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){
            return std::tolower(c);
        });

        return str;
    }

    void CommandLoop() {
        // mlp::Dataset train_dataset("res/mnist/mnist_train.csv");
        // mlp::Dataset test_dataset("res/mnist/mnist_test.csv");

        // mlp::Dataset dataset("res/testing/test_data.csv");
        mlp::Dataset dataset("res/testing/test_data_2.csv");

        mlp::cuda::Profiler cuda_profiler;
        mlp::host::Profiler host_profiler;

        mlp::cuda::Context cuda_context(&cuda_profiler);
        mlp::host::Context host_context(&host_profiler);

        constexpr size_t BATCH_SIZE = 4;

        // std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };
        std::vector<size_t> layer_sizes = { 4, 4, 4 };

        mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);
        // train_dataset.parseBatch(batch);
        // test_dataset.parseBatch(batch);
        dataset.parseBatch(batch);

        mlp::MLP<host::Context> model(host_context, BATCH_SIZE);
        model.init(layer_sizes);

        model.forwardPass(batch);
        host_profiler.print();

        float* host_outputs = new float[BATCH_SIZE * layer_sizes.back()];

        model.copyOutputs(host_outputs);

        for (size_t i = 0; i < BATCH_SIZE * layer_sizes.back(); i++) {
            std::cout << "  " << host_outputs[i];
        }
        std::cout << "\n";

        delete[] host_outputs;

        // for (int i = 0; i < 100; i++) {
        //     // dataset.parseBatch(batch);
        //     test_dataset.parseBatch(batch);
        //     model.forwardPass(batch);
        //     // model.backwardPass(batch);
        // }

        // Test(model, dataset);

        std::string command;

        while (true) {
            std::getline(std::cin, command);

            std::istringstream iss(command);
            std::vector<std::string> args;
            std::string arg;

            while (iss >> arg) {
                args.emplace_back(to_lower(arg));
            }

            if (args.size() < 1) continue;

            std::string& cmd = args[0];

            if (cmd == "exit" || cmd == "quit") {
                break;
            }
            else if (cmd == "help") {
            }
            else {
                std::cout << "Unknown command: '" << cmd << "'\n";
            }
        }
    }
}