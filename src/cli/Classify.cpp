#include "cli/CLI.hpp"
#include "cli/Commands.hpp"

namespace mlp {
    namespace commands {
        void classify_sample(MLP_t* model, const ImageData& image_data, const int label) {
            Sample sample(image_data.data.size());
            sample.label = label;

            std::memcpy(sample.data.data(), image_data.data.data(), sample.data.size() * sizeof(float));

            ClassifyInfo info;

            model->classify(sample, info);

            std::cout << "\nClassification:   " << info.classification << "\n\n";

            for (size_t i = 0; i < info.outputs.size(); i++) {
                std::cout << i << ": " << info.outputs[i] << "\n";
            }

            if (info.error == UNDEFINED_ERROR) return;

            std::cout << "\nError/loss:       " << info.error << "\n\n";
        }
    }
}