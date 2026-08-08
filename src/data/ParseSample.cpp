#include <fstream>
#include <iostream>
#include <sstream>

#include "data/ParseSample.hpp"

namespace mlp {
    bool parse_sample(const std::string& path, ImageData& sample) {
        std::ifstream input(path);

        if (!input.is_open()) {
            std::cout << "Could not open: " << path << "\n";
            return false;
        }

        std::stringstream content;
        content << input.rdbuf();
        input.close();

        std::string content_str = content.str();

        size_t index = 0;
        int current_value = 0;
        bool reading_number = false;

        std::vector<int> dimensions;

        for (char c : content_str) {
            if (std::isdigit(c)) {
                reading_number = true;
                current_value *= 10;
                current_value += c - '0';
                continue;
            }

            if (!reading_number) continue;
            reading_number = false;

            if (dimensions.size() < 2) dimensions.push_back(current_value);
            else sample.data.push_back((float) current_value / 255.0f);

            current_value = 0;
        }

        sample.width = dimensions.at(0);
        sample.height = dimensions.at(1);

        return true;
    }
}