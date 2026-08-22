#pragma once

#include <string>
#include <vector>

namespace mlp {
    struct ImageData {
        int width = 0;
        int height = 0;
        std::vector<float> data;
    };

    bool parse_sample(const std::string& path, ImageData& sample, const size_t max_size = 10000);
}