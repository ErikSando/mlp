#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
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

        mlp::Dataset test_ds("res/testing/test_data.csv");

        mlp::Profiler profiler;

        mlp::Context context(&profiler);

        constexpr size_t BATCH_SIZE = 32;
        // constexpr size_t BATCH_SIZE = 4;
        constexpr float LEARNING_RATE = 0.01f;

        std::unordered_map<std::string, Activation> activation_functions; // maybe seperate hidden and output activation functions
        std::unordered_map<std::string, Loss> loss_functions;

        activation_functions["none"] = Activation::NONE;
        activation_functions["relu"] = Activation::RELU;
        activation_functions["leakyrelu"] = Activation::LEAKY_RELU;
        activation_functions["sigmoid"] = Activation::SIGMOID;
        activation_functions["tanh"] = Activation::TANH;
        activation_functions["softmax"] = Activation::SOFTMAX;

        loss_functions["cce"] = Loss::CCE;
        loss_functions["mse"] = Loss::MSE;

        std::unordered_map<std::string, mlp::MLP_t> models;

        // to do: check for a base model save in res/ and load it into the base model if it exists

        std::string BASE_NAME = "base";

        // base model
        std::vector<size_t> layer_sizes = { 784, 128, 64, 10 };
        // std::vector<size_t> layer_sizes = { 100, 32, 4 };
        auto [it, _] = models.emplace(BASE_NAME, context);
        it->second.init(layer_sizes, BATCH_SIZE);

        mlp::MLP_t* model = &models.at(BASE_NAME);
        std::string model_name = BASE_NAME;

        // mlp::Batch batch(BATCH_SIZE, layer_sizes[0]);

        // TestData test_data;

        // std::cout << "testing...\n";

        // commands::test(model, context, mnist_test_ds, test_data);

        // std::cout << "Accuracy: " << (test_data.getAccuracy() * 100) << "% (" << test_data.correct << "/" << test_data.correct + test_data.incorrect << ")\n";
        // std::cout << "backward passing...\n";

        // // for (size_t i = 0; i < mnist_train_ds.size() * 2 / BATCH_SIZE; i++) {
        // for (size_t i = 0; i < 400; i++) {
        //     mnist_train_ds.parseBatch(batch);
        //     model->forwardPass(batch);
        //     model->backwardPass(batch);
        // }

        // std::cout << "done.\n";
        // std::cout << "synchronising...\n";

        // context.synchronise();

        // std::cout << "done.\n";
        // std::cout << "testing...\n";

        // commands::test(model, context, mnist_test_ds, test_data);

        // std::cout << "Accuracy: " << (test_data.getAccuracy() * 100) << "% (" << test_data.correct << "/" << test_data.correct + test_data.incorrect << ")\n";

        std::string command;

        while (true) {
            std::cout << model_name << " > ";
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
                std::cout << "* COMMANDS HERE ARE A TODO LIST, NOT MANY DO ANYTHING YET\n";
                std::cout << "\nhelp\n - Displays this menu.\n";
                std::cout << "\nbuild [name] [no. inputs] [hidden layer sizes] [no. outputs] [hidden activiation] [output activation] [loss] [batch size: optional] [learning rate: optional]\
                \n - Build a model with the given parameters. For hidden layer sizes, give values seperated by commas e.g. 128,64 (without spaces). For no hidden layers leave any non-number character. \
                If no value is given, the default batch size is 32, and the default learning rate is 0.02.\
                \n Hidden activation functions:\n - none, relu, leakyrelu, sigmoid, tanh.\
                \n Output activation functions:\n - none, softmax.\
                \n Loss functions:\n - cce, mse.\n";
                std::cout << "\nactiviate [name]\n - Set the model with the given name as the current model. Any commands ran will use this model.";
                std::cout << "\train [no. epochs] [dataset file path]\n - Train the model with the specified number of epochs, using the specified dataset file.\n";
                std::cout << "\ntest [dataset file path] [dataset file path]\n - Test the model's accuracy over one epoch, using the specified dataset file.\n";
                std::cout << "\nprint [image data path]\n - Visualise the data in a file in the terminal.\n";
                std::cout << "\nclassify [image data path] [label: optional]\n - Use the network to classify an image. If a label is given, the error/loss will be printed.\n - Aliases: class, id, identify.\n";
                std::cout << "\nsave [save path]\n - Save the model data to a file. This includes the name, layer sizes, activation functions, loss function, batch size, and learning rate.\n";
                std::cout << "\nload [save path]\n - Build a new model using the data from the save file. If a model with the same name already exists, it will not be overwritten.\n";
                std::cout << "\nexit\n - Terminate the program.\n - Aliases: quit.\n\n";
            }
            else if (cmd == "build") {
                if (args.size() < 8) {
                    std::cout << "Insufficient arguments" << std::endl;
                    std::cout << "Usage: build [name] [no. inputs] [hidden layer sizes] [no. outputs] [hidden activiation] [output activation] [loss] [batch size: optional] [learning rate: optional]" << std::endl;
                    continue;
                }

                std::string& name = args.at(1);

                if (models.find(name) != models.end()) {
                    std::cout << "Model with name '" << name << "' already exists, can not create a new model with the same name.\n";
                    continue;
                }

                std::vector<size_t> layer_sizes;

                layer_sizes.push_back(std::stoi(args.at(2)));

                std::string& hidden_sizes = args.at(3);

                std::vector<size_t> split_indices_left = { 0 };
                std::vector<size_t> split_indices_right;

                for (size_t i = 0; i < hidden_sizes.size(); i++) {
                    if (hidden_sizes[i] == ',') {
                        split_indices_right.push_back(i > 0 ? i - 1 : 0);
                        split_indices_left.push_back(i + 1);
                    }
                }

                if (split_indices_right.size() == 0) split_indices_right.emplace_back(hidden_sizes.size() - 1);

                // for (size_t i = 0; i < split_indices_left.size(); i++) {
                //     std::cout << split_indices_left[i] << ",";
                //     if (i < split_indices_right.size()) {
                //         std::cout << split_indices_right[i];
                //     }
                //     std::cout << "\n";
                // }

                for (size_t i = 0; i < split_indices_left.size(); i++) {
                    size_t left = split_indices_left[i];
                    size_t right = i < split_indices_right.size() ? split_indices_right[i] : hidden_sizes.size() - 1;

                    if (left >= right) continue;

                    size_t layer_size = 0;

                    for (size_t j = 0; j <= right - left; j++) {
                        char digit = hidden_sizes[left + j];
                        layer_size *= 10;

                        size_t next_digit = (size_t) (digit - '0');
                        if (next_digit > 9) {
                            // throw std::runtime_error("Invalid layer size given");
                            layer_size = 0;
                            break;
                        }
                        layer_size += next_digit;
                    }

                    if (layer_size > 0) layer_sizes.push_back(layer_size);
                }

                layer_sizes.push_back(std::stoi(args.at(4)));

                // for (size_t size : layer_sizes) {
                //     std::cout << size << "\n";
                // }

                std::string hidden_activation_str = to_lower(args.at(5));
                std::string output_activation_str = to_lower(args.at(6));
                std::string loss_function_str = to_lower(args.at(7));

                if (activation_functions.find(hidden_activation_str) == activation_functions.end()) {
                    std::cerr << "Invalid hidden layer activation function: '" << hidden_activation_str << "'\n";
                    continue;
                }

                if (activation_functions.find(output_activation_str) == activation_functions.end()) {
                    std::cerr << "Invalid output layer activation function: '" << output_activation_str << "'\n";
                    continue;
                }

                if (loss_functions.find(loss_function_str) == loss_functions.end()) {
                    std::cerr << "Invalid loss function: '" << loss_function_str << "'\n";
                    continue;
                }

                Activation hidden_activation = activation_functions.at(hidden_activation_str);
                Activation output_activation = activation_functions.at(output_activation_str);
                Loss loss_function = loss_functions.at(loss_function_str);

                size_t batch_size = args.size() > 8 ? (size_t) std::stoi(args.at(8)) : BATCH_SIZE;
                float learning_rate = args.size() > 9 ? std::stof(args.at(9)) : LEARNING_RATE;

                // auto [it, inserted] = models.emplace(name, context);

                // if (!inserted) {
                //     std::cout << "Model with name '" << name << "' already exists, can not create a new model with the same name.\n";
                //     continue;
                // }

                auto [it, _] = models.emplace(name, context);

                it->second.init(layer_sizes, batch_size, hidden_activation, output_activation, loss_function, learning_rate);

                std::cout << "Created model with name '" << name << "'\n";
                std::cout << "Layer sizes: " << layer_sizes[0];
                for (size_t i = 1; i < layer_sizes.size(); i++) std::cout << ", " << layer_sizes[i];
                std::cout << "\n";
                std::cout << "Hidden activation function: " << hidden_activation_str << "\n";
                std::cout << "Output activation function: " << output_activation_str << "\n";
                std::cout << "Loss function: " << loss_function_str << "\n";
                std::cout << "Batch size: " << batch_size << "\n";
                std::cout << "Learning rate: " << learning_rate << "\n";
            }
            else if (cmd == "activate") {
                if (args.size() < 2) {
                    std::cout << "Insufficient arguments" << std::endl;
                    std::cout << "Usage: activate [name]" << std::endl;
                    continue;
                }

                std::string& name = args.at(1);

                if (models.find(name) == models.end()) {
                    std::cout << "Can not find model with name '" << name << "'\n";
                    continue;
                }

                model = &models.at(name);
                model_name = name;
                std::cout << "Activated model '" << name << "'\n";
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
            else if (cmd == "id" || cmd == "class" || cmd == "classify" || cmd == "identify") {
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
            else if (cmd == "train") {
                if (args.size() < 3) {
                    std::cout << "Insufficient arguments" << std::endl;
                    std::cout << "Usage: train [no. epochs] [dataset file path]" << std::endl;
                    continue;
                }

                size_t n_epochs = static_cast<size_t>(std::stoi(args.at(1)));
                std::string& ds_path = args.at(2);

                mlp::Dataset dataset(ds_path);

                if (!dataset.isSetup()) {
                    std::cout << "Failed to create dataset using path '" << ds_path << "'\n";
                    continue;
                }

                std::cout << "Training with " << n_epochs << " epoch/s...\n";

                commands::train(model, dataset, n_epochs);
            }
            else if (cmd == "test") {
                if (args.size() < 2) {
                    std::cout << "Insufficient arguments" << std::endl;
                    std::cout << "Usage: test [dataset file path]" << std::endl;
                    continue;
                }

                std::string& ds_path = args.at(1);

                mlp::Dataset dataset(ds_path);

                if (!dataset.isSetup()) {
                    std::cout << "Failed to create dataset using path '" << ds_path << "'\n";
                    continue;
                }

                TestData test_data;
                commands::test(model, context, dataset, test_data);

                std::cout << "Accuracy: " << (test_data.getAccuracy() * 100) << "% (" << test_data.correct << "/" << test_data.correct + test_data.incorrect << ")\n";
            }
            else {
                std::cout << "Unknown command: '" << cmd << "'\n";
            }
        }
    }
}