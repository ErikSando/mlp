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
        // bool save(MLP_t* model, std::string save_path) {
        //     std::ofstream file(save_path);

        //     if (!file.is_open()) {
        //         std::cerr << "Error: failed to open file '" << save_path << "'\n";
        //         return false;
        //     }

        //     const Context& context = model->getContext();
        //     const std::vector<MLP_t::Layer_t_up>& layers = model->getLayers();

        //     ModelData data;
        //     model->getData(data);

        //     for (size_t layer_size : data.layer_sizes) {
        //         file << layer_size << ",";
        //     }
        //     file << "\n";

        //     file << activation_string(data.hidden_activation) << "," << activation_string(data.output_activation) << "\n";

        //     file << loss_string(data.loss_function) << "\n";
        //     file << data.batch_size << "\n";
        //     file << data.learning_rate << "\n";

        //     for (size_t i = 1; i < layers.size(); i++) { // input layer doesn't have weights, so start at index 1 to skip it
        //         const MLP_t::Layer_t_up& layer = layers.at(i);
        //         std::vector<float> host_weights(layer->weights.size());
        //         std::vector<float> host_biases(layer->biases.size());

        //         context.transfer(host_weights.data(), layer->weights);
        //         context.transfer(host_biases.data(), layer->biases);

        //         for (float weight : host_weights) {
        //             file << weight << ",";
        //         }
        //         file << "\n";

        //         for (float bias : host_biases) {
        //             file << bias << ",";
        //         }
        //         file << "\n";
        //     }

        //     file.close();

        //     return true;
        // }

        bool save(const ModelData& data, std::string save_path) {
            std::ofstream file(save_path);

            if (!file.is_open()) {
                std::cerr << "Error: failed to open file '" << save_path << "'\n";
                return false;
            }

            for (size_t layer_size : data.layer_sizes) {
                file << layer_size << ",";
            }
            file << "\n";

            file << activation_string(data.hidden_activation) << "," << activation_string(data.output_activation) << "\n";

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
                    file << weight << ",";
                }
                file << "\n";

                for (float bias : biases) {
                    file << bias << ",";
                }
                file << "\n";
            }

            file.close();

            return true;
        }
    }
}