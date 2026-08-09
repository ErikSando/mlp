#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli/CLI.hpp"
#include "cli/Commands.hpp"
#include "cuda/Context.hpp"
#include "cuda/profiling/Profiler.hpp"
#include "data/Dataset.hpp"
#include "data/ParseSample.hpp"
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

    void command_loop() {
        mlp::Dataset mnist_train_ds("res/mnist/mnist_train.csv");
        mlp::Dataset mnist_test_ds("res/mnist/mnist_test.csv");

        mlp::cuda::Profiler cuda_profiler;
        mlp::host::Profiler host_profiler;

        mlp::cuda::Context cuda_context(&cuda_profiler);
        mlp::host::Context host_context(&host_profiler);

        constexpr size_t BATCH_SIZE = 32;

        std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };
        // std::vector<size_t> layer_sizes = { 4, 4, 4 };

        mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);
        // train_dataset.parseBatch(batch);
        // test_dataset.parseBatch(batch);
        mnist_train_ds.parseBatch(batch);

        mlp::MLP<cuda::Context> model(cuda_context); // there is a problem with the CUDA context right now (possibly something to do with Layer or Matrix deletion/copying/moving idk)
        model.init(layer_sizes, BATCH_SIZE);

        // model.forwardPass(batch);

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
                std::cout << "\nhelp:\n - Displays this menu.\n";
                std::cout << "\nprint [image data path]:\n - Visualise the data in a file in the terminal.\n";
                std::cout << "\nclassify [image data path] [label: optional]\n - Use the network to classify an image. If a label is given, the error/loss will be printed.\n - Aliases: class, id, identify.\n";
                std::cout << "\nexit:\n - Terminate the program.\n - Aliases: quit.\n\n";
            }
            else if (cmd == "print") {
                if (args.size() < 2) {
                    std::cout << "Insufficient arguments" << std::endl;
                    std::cout << "Usage: print [image data path]" << std::endl;
                    continue;
                }

                std::string path = args.at(1);

                ImageData sample;

                if (!parse_sample(path, sample)) {
                    std::cout << "Failed to parse sample\n";
                    continue;
                }

                commands::print_sample(sample);
            }
            else if (cmd == "id" || cmd == "class" || cmd == "classify" || "identify") {
                if (args.size() < 2) {
                    std::cout << "Insufficient arguments" << std::endl;
                    std::cout << "Usage: " << cmd << " [image data path] [digit: optional]" << std::endl;
                    continue;
                }

                std::string path = args.at(1);

                bool label_specified = false;
                int label = NO_LABEL;

                if (args.size() >= 3) label = std::stoi(args.at(2));

                ImageData sample;

                if (!parse_sample(path, sample)) {
                    std::cout << "Failed to parse sample\n";
                    continue;
                }

                commands::classify_sample(model, sample, label);
            }
            else {
                std::cout << "Unknown command: '" << cmd << "'\n";
            }
        }
    }
}