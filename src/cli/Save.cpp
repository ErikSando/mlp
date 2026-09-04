#include <fstream>

#include "cli/CLI.hpp"
#include "cli/Commands.hpp"

namespace mlp {
    std::string activation_string(const Activation activation) {
        switch (activation) {
            case Activation::NONE:         return "none"; break;
            case Activation::LEAKY_RELU:   return "leakyrelu"; break;
            case Activation::RELU:         return "relu"; break;
            case Activation::SIGMOID:      return "sigmoid"; break;
            case Activation::TANH:         return "tanh"; break;
            case Activation::SOFTMAX:      return "softmax"; break;
            default:
                throw std::runtime_error("Invalid activation function");
            break;
        }
    }

    std::string loss_string(const Loss loss) {
        switch (loss) {
            case Loss::CCE:     return "cce"; break;
            case Loss::MSE:     return "mse"; break;
            default:
                throw std::runtime_error("Invalid loss function");
            break;
        }
    }

    namespace commands {
        bool save(const ModelData& data, std::string save_path) {
            std::ofstream file(save_path);

            if (!file.is_open()) {
                std::cerr << "Error: failed to open file '" << save_path << "'\n";
                return false;
            }

            for (size_t layer_size : data.layer_sizes) {
                file << layer_size << seperator;
            }
            file << "\n";

            file << activation_string(data.hidden_activation) << seperator << activation_string(data.output_activation) << "\n";

            file << loss_string(data.loss_function) << "\n";
            file << data.batch_size << "\n";
            file << data.learning_rate << "\n";

            if (data.weights.size() != data.biases.size()) {
                std::cerr << "Error: mismatch between number of weight matrices and number of bias vectors\n";
                return false;
            }

            for (size_t i = 0; i < data.weights.size(); i++) {
                const std::vector<float>& weights = data.weights.at(i);
                const std::vector<float>& biases = data.biases.at(i);

                for (float weight : weights) {
                    file << weight << seperator;
                }
                file << "\n";

                for (float bias : biases) {
                    file << bias << seperator;
                }
                file << "\n";
            }

            file.close();

            return true;
        }
    }
}