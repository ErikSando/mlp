#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>
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

    // default values
    constexpr size_t BATCH_SIZE = 32UL;
    constexpr float LEARNING_RATE = 0.01f;

    std::unordered_map<std::string, Activation> activation_functions; // maybe seperate hidden and output activation functions
    std::unordered_map<std::string, Loss> loss_functions;

    void command_loop() {
        mlp::Dataset mnist_train_ds("res/mnist/mnist_train.csv");
        mlp::Dataset mnist_test_ds("res/mnist/mnist_test.csv");

        // mlp::Dataset test_ds("res/testing/test_data_2.csv");

        mlp::Profiler profiler(PROFILER_NAME);

        mlp::Context context(&profiler);

        activation_functions["none"] = Activation::NONE;
        activation_functions["leakyrelu"] = Activation::LEAKY_RELU;
        activation_functions["relu"] = Activation::RELU;
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
        auto [it, _] = models.emplace(BASE_NAME, context);
        it->second.init(layer_sizes);

        mlp::MLP_t* model = &models.at(BASE_NAME);
        std::string model_name = BASE_NAME;

        bool print_profile = false; // maybe i can just rely on the enabled member bool in the profiler class

        std::string command;

        profiler.clear();

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
                std::cout << "\nhelp\n - Displays this menu.\n";
                std::cout << "\nbuild [name] [no. inputs] [hidden layer sizes] [no. outputs] [hidden activiation] [output activation] [loss] [batch size: optional] [learning rate: optional]\
\n - Build a model with the given parameters. For hidden layer sizes, give values seperated by commas e.g. 128,64 (without spaces). For no hidden layers leave any non-number character. \
If no value is given, the default batch size is 32, and the default learning rate is 0.01.\
\n Hidden activation functions:\n - none, relu, leakyrelu, sigmoid, tanh.\
\n Output activation functions:\n - none, softmax.\
\n Loss functions:\n - cce, mse.\n";
                std::cout << "\nactiviate [name]\n - Set the model with the given name as the current model. Any commands ran will use this model.\n";
                std::cout << "\ninfo\n - Display information about the model: layer sizes, activation functions, loss function, batch size, and learning rate.\n";
                std::cout << "\train [no. epochs] [dataset file path]\n - Train the model with the specified number of epochs, using the specified dataset file.\n";
                std::cout << "\ntest [dataset file path] [dataset file path]\n - Test the model's accuracy over one epoch, using the specified dataset file.\n";
                std::cout << "\nprint [image data path]\n - Visualise the data in a file in the terminal.\n";
                std::cout << "\nclassify [image data path] [label: optional]\n - Use the network to classify an image. If a label is given, the error/loss will be printed.\n - Aliases: class, id, identify.\n";
                std::cout << "\nprofile [optional: value]\n - Entering this command with no arguments will print the time spent on tasks tracked by the profiler during the last use of the test or train command.\
                \n   If an argument is given (e.g. true, false), it will be used to set whether the profiler automatically prints after the test and train commands.\n";
                std::cout << "\nsave [save path]\n - Save the model data to a file. This includes the name, layer sizes, activation functions, loss function, batch size, and learning rate.\n";
                std::cout << "\nload [model name] [save path]\n - Build a new model with the given name using the data from the save file. If a model with the same name already exists, it will not be overwritten.\n";
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

                auto [it, _] = models.emplace(name, context);

                // it is already checked that a model with the same name does not exist
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
            else if (cmd == "info") {
                ModelData data;
                model->exportData(data);

                std::cout << "Layer sizes: " << data.layer_sizes.at(0);

                for (size_t i = 1; i < data.layer_sizes.size(); i++) {
                    std::cout << ", " << data.layer_sizes.at(i);
                }
                std::cout << "\n";

                std::cout << "Hidden layer count: " << data.layer_sizes.size() - 2 << "\n";
                std::cout << "Hidden layer activation function: " << activation_string(data.hidden_activation) << "\n";
                std::cout << "Output layer activation function: " << activation_string(data.output_activation) << "\n";
                std::cout << "Loss function: " << loss_string(data.loss_function) << "\n";
                std::cout << "Batch size: " << data.batch_size << "\n";
                std::cout << "Learning rate: " << data.learning_rate << "\n";
            }
            else if (cmd == "print") {
                if (args.size() < 2) {
                    std::cout << "\nInsufficient arguments\n";
                    std::cout << "Usage: print [image data path]\n\n";
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
                    std::cout << "\nInsufficient arguments\n";
                    std::cout << "Usage: " << cmd << " [image data path] [digit: optional]\n\n";
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

                bool enable = profiler.enabled();
                profiler.disable();

                commands::classify_sample(model, sample, label);

                if (enable) profiler.enable();
            }
            else if (cmd == "train") {
                if (args.size() < 3) {
                    std::cout << "\nInsufficient arguments\n";
                    std::cout << "Usage: train [no. epochs] [dataset file path]\n\n";
                    continue;
                }

                size_t n_epochs = static_cast<size_t>(std::stoi(args.at(1)));
                std::string& ds_path = args.at(2);

                mlp::Dataset dataset(ds_path);

                if (!dataset.isSetup()) {
                    std::cout << "\nFailed to create dataset using path '" << ds_path << "'\n\n";
                    continue;
                }

                std::cout << "\nTraining with " << n_epochs << " epoch/s...\n";

                profiler.clear();
                profiler.startBenchmark();

                commands::train(model, dataset, n_epochs);

                profiler.endBenchmark();

                std::cout << "Training completed\n\n";

                if (print_profile) {
                    profiler.print();
                    std::cout << '\n';
                }
            }
            else if (cmd == "test") {
                if (args.size() < 2) {
                    std::cout << "\nInsufficient arguments\n";
                    std::cout << "Usage: test [dataset file path]\n\n";
                    continue;
                }

                std::string& ds_path = args.at(1);

                mlp::Dataset dataset(ds_path);

                if (!dataset.isSetup()) {
                    std::cout << "\nFailed to create dataset using path '" << ds_path << "'\n\n";
                    continue;
                }

                profiler.clear();
                profiler.startBenchmark();

                TestData test_data;
                commands::test(model, dataset, test_data);

                profiler.endBenchmark();

                if (print_profile) {
                    std::cout << '\n';
                    profiler.print();
                }

                std::cout << "\nAccuracy: " << (test_data.getAccuracy() * 100) << "% (" << test_data.correct << "/" << test_data.correct + test_data.incorrect << ")\n\n";
            }
            else if (cmd == "profile") {
                if (args.size() < 2) {
                    std::cout << '\n';
                    profiler.print();
                    std::cout << '\n';
                    continue;
                }

                std::string& value = args.at(1);

                if (value == "true" || value == "t" || value == "yes" || value == "y") {
                    print_profile = true;
                    std::cout << "Profiler will automatically print\n";
                    continue;
                }

                if (value == "false" || value == "f" || value == "no" || value == "n") {
                    print_profile = false;
                    std::cout << "Profiler will not automatically print\n";
                    continue;
                }

                std::cout << "Invalid option: '" << args.at(1) << "'\n";
                std::cout << "Use one out of: true, false, yes, no, t, f, y, n\n";
            }
            else if (cmd == "save") {
                if (args.size() < 2) {
                    std::cout << "\nInsufficient arguments\n";
                    std::cout << "Usage: save [save path]\n\n";
                    continue;
                }

                std::string& save_path = args.at(1);

                ModelData data;
                model->exportData(data);

                bool enable = profiler.enabled();
                profiler.disable();

                commands::save(data, save_path);

                if (enable) profiler.enable();
            }
            else if (cmd == "load") {
                if (args.size() < 3) {
                    std::cout << "\nInsufficient arguments\n";
                    std::cout << "Usage: load [model name] [save path]\n\n";
                    continue;
                }

                std::string& name = args.at(1);
                std::string& save_path = args.at(2);

                if (models.find(name) != models.end()) {
                    std::cout << "Model with name '" << name << "' already exists, can not create a new model with the same name.\n";
                    continue;
                }

                ModelData data;

                bool enable = profiler.enabled();
                profiler.disable();

                bool load_success = commands::load(data, save_path);

                if (enable) profiler.enable();

                if (!load_success) {
                    std::cout << "Failed to load model data from file, can not build model.\n";
                    continue;
                }

                auto [it, _] = models.emplace(name, context);

                it->second.init(data);
            }
            else {
                std::cout << "\nUnknown command: '" << cmd << "'\n\n";
            }
        }
    }
}