#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli/CLI.hpp"
#include "cli/Commands.hpp"
#include "data/Dataset.hpp"
#include "data/ParseSample.hpp"
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

        mlp::Profiler profiler;

        mlp::Context context(&profiler);

        constexpr size_t BATCH_SIZE = 1;

        std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };

        mlp::MLP_t model(context);
        model.init(layer_sizes, BATCH_SIZE);

        mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);

        // std::cout << "backward passing...\n";

        // for (size_t i = 0; i < 10000; i++) {
        //     mnist_train_ds.parseBatch(batch);
        //     model.forwardPass(batch);
        //     model.backwardPass(batch);
        // }

        // std::cout << "synchronising...\n";

        // context.synchronise();

        // std::cout << "done.\n";

        TestData test_data;

        commands::test(model, context, mnist_test_ds, test_data);

        std::cout << "Accuracy: " << (test_data.getAccuracy() * 100) << "% (" << test_data.correct << " correct)\n";

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
                std::cout << "\nhelp\n - Displays this menu.\n";
                std::cout << "\nprint [image data path]\n - Visualise the data in a file in the terminal.\n";
                std::cout << "\nclassify [image data path] [label: optional]\n - Use the network to classify an image. If a label is given, the error/loss will be printed.\n - Aliases: class, id, identify.\n";
                std::cout << "\ntest\n - Test the model's accuracy over one epoch.\n";
                std::cout << "\nexit\n - Terminate the program.\n - Aliases: quit.\n\n";
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
                    std::cout << "Failed to parse sample.\n";
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
                    std::cout << "Failed to parse sample.\n";
                    continue;
                }

                commands::classify_sample(model, sample, label);
            }
            else if (cmd == "test") {
                
            }
            else {
                std::cout << "Unknown command: '" << cmd << "'\n";
            }
        }
    }
}