#include <fstream>

#include "cli/CLI.hpp"
#include "cli/Commands.hpp"

namespace mlp {
    namespace commands {
        bool read_floats(std::string& line, std::vector<float>& values, std::string type) {
            std::string value_str = "";

            for (size_t i = 0; i < line.size(); i++) {
                char c = line[i];

                if (std::isdigit(c) || c == '-' || c == '.' || c == 'e') {
                    value_str += c;
                    continue;
                }

                if (!value_str.size()) {
                    std::cerr << "Error: exported " << type << " is an empty string\n";
                    return false;
                }

                values.push_back(std::stof(value_str));
                value_str = "";
            }

            return true;
        }

        bool load(ModelData& data, std::string save_path) {
            std::ifstream file(save_path);

            if (!file.is_open()) {
                std::cerr << "Error: failed to open file '" << save_path << "'\n";
                return false;
            }

            std::string line;

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no layer sizes found\n";
                return false;
            }

            data.layer_sizes.clear();

            size_t value = 0;

            for (size_t i = 0; i < line.size(); i++) {
                char c = line[i];

                if (std::isdigit(c)) {
                    value *= 10;
                    value += (size_t) (c - '0');
                    continue;
                }

                if (value == 0) {
                    std::cerr << "Error: layer size of zero is not allowed\n";
                    return false;
                }

                data.layer_sizes.push_back(value);
                value = 0;
            }

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no activation functions found\n";
                return false;
            }

            size_t comma_pos = line.find(seperator);

            if (comma_pos >= line.size()) {
                std::cerr << "Error: no comma found in activation functions line\n";
                return false;
            }

            std::string h_activ_str = line.substr(0, comma_pos);
            std::string o_activ_str = line.substr(comma_pos + 1, line.size() - comma_pos - 1);

            auto it_h = activation_functions.find(h_activ_str);
            auto it_o = activation_functions.find(o_activ_str);

            if (it_h == activation_functions.end()) {
                std::cerr << "Error: invalid activation function '" << h_activ_str << "'\n";
                return false;
            }

            if (it_o == activation_functions.end()) {
                std::cerr << "Error: invalid activation function '" << o_activ_str << "'\n";
                return false;
            }

            data.hidden_activation = it_h->second;
            data.output_activation = it_o->second;

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no loss function found\n";
                return false;
            }

            auto it = loss_functions.find(line);

            if (it == loss_functions.end()) {
                std::cerr << "Error: invalid loss function '" << line << "'\n";
                return false;
            }

            data.loss_function = it->second;

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no batch size found\n";
                return false;
            }

            data.batch_size = static_cast<size_t>(std::stoi(line));

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no learning rate found\n";
                return false;
            }

            data.learning_rate = std::stof(line);

            size_t n_layers = data.layer_sizes.size() - 1;

            std::string weights_line;
            std::string biases_line;

            for (size_t i = 0; i < n_layers; i++) {
                if (!std::getline(file, weights_line)) {
                    std::cerr << "Error: missing line for weights\n";
                    return false;
                }

                if (!std::getline(file, biases_line)) {
                    std::cerr << "Error: missing line for biases\n";
                    return false;
                }

                data.weights.emplace_back();
                data.biases.emplace_back();

                if (!read_floats(weights_line, data.weights.back(), "weights")) return false;
                if (!read_floats(biases_line, data.biases.back(), "biases")) return false;

                if (data.weights.back().size() != data.layer_sizes.at(i) * data.layer_sizes.at(i + 1)) {
                    std::cout << "Error: incorrect number of weights, expected " << data.layer_sizes.at(i) * data.layer_sizes.at(i + 1) << ", got " << data.weights.back().size() << "\n";
                    return false;
                }

                if (data.biases.back().size() != data.layer_sizes.at(i + 1)) {
                    std::cout << "Error: incorrect number of biases, expected " << data.layer_sizes.at(i + 1) << ", got " << data.biases.back().size() << "\n";
                    return false;
                }
            }

            file.close();

            return true;
        }
    }
}