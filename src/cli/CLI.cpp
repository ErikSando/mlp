#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli/CLI.hpp"
#include "data/Dataset.hpp"
#include "device/CUDAContext.hpp"
#include "mlp/MLP.hpp"
#include "profiling/Profiler.hpp"

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

        mlp::CUDAProfiler cuda_profiler;
        mlp::Profiler profiler;

        mlp::CUDAContext context(&cuda_profiler);

        constexpr size_t BATCH_SIZE = 1;

        // std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };
        std::vector<size_t> layer_sizes = { 4, 4, 4 };

        mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);
        // train_dataset.parseBatch(batch);
        dataset.parseBatch(batch);

        mlp::MLP<CUDAContext> model(context, BATCH_SIZE);
        model.init(layer_sizes);

        for (int i = 0; i < 10; i++) {
            dataset.parseBatch(batch);
            model.forwardPass(batch);
            model.backwardPass(batch);
        }

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