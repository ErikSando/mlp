#pragma once

#include <cstring>

#include "data/ParseSample.hpp"
#include "data/Sample.hpp"
#include "mlp/MLP.hpp"

namespace mlp {
    namespace commands {
        void print_sample(const ImageData& sample);

        template<typename TContext>
        void classify_sample(MLP<TContext>& model, const ImageData& image_data, const int label = NO_LABEL) {
            Sample sample(image_data.data.size());
            sample.label = label;

            std::memcpy(sample.data.data(), image_data.data.data(), sample.data.size() * sizeof(float));

            ClassifyInfo info;

            model.classify(sample, info);

            std::cout << "Classification:   " << info.classification << "\n";

            for (size_t i = 0; i < info.outputs.size(); i++) {
                std::cout << i << ": " << info.outputs[i] << "\n";
            }

            if (info.error == UNDEFINED_ERROR) return;

            std::cout << "Error/loss:       " << info.error << "\n";
        }
    }
}