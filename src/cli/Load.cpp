#include <fstream>

#include "cli/CLI.hpp"
#include "cli/Commands.hpp"

namespace mlp {
    namespace commands {
        bool load(ModelData& data, std::string save_path) {
            std::ifstream file(save_path);

            if (!file.is_open()) {
                std::cerr << "Error: failed to open file '" << save_path << "'\n";
                return false;
            }

            // const Context& context = model->getContext();
            // const std::vector<MLP_t::Layer_t_up>& layers = model->getLayers();

            std::string line;

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no layer sizes found\n";
                return false;
            }

            data.layer_sizes.clear();

            size_t value = 0;

            for (size_t i = 0; i < line.size(); i++) {
                char c = line[i];

                if (!std::isdigit(c)) {
                    if (value == 0) {
                        std::cerr << "Error: layer size of zero is not allowed\n";
                        return false;
                    }

                    data.layer_sizes.push_back(value);
                    value = 0;
                    continue;
                }

                value *= 10;
                value += (size_t) (c - '0');
            }



            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no activation functions found\n";
                return false;
            }

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no loss function found\n";
                return false;
            }

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no batch size found\n";
                return false;
            }

            if (!std::getline(file, line)) {
                std::cerr << "Error: insufficient lines in save file, no learning rate found\n";
                return false;
            }

            // then go through the lines for weights and biases

            file.close();

            return true;
        }
    }
}