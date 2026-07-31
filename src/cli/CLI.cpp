#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli/CLI.hpp"
#include "data/Dataset.hpp"
#include "device/DeviceContext.hpp"
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
        mlp::Dataset dataset("res/mnist/mnist_train.csv");

        mlp::CUDAProfiler cuda_profiler;
        mlp::Profiler profiler;

        mlp::DeviceContext context(&cuda_profiler);

        constexpr size_t BATCH_SIZE = 32;

        std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };

        mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);
        dataset.parseBatch(batch);

        mlp::MLP model(context, BATCH_SIZE);
        model.init(layer_sizes);
        model.forwardPass(batch);
        model.backwardPass(batch);

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