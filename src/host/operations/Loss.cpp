#include <cmath>

#include "enums/Enums.hpp"
#include "host/Context.hpp"

namespace mlp {
    namespace host {
        // I'm using the same type of kernel approach as with CUDA even though it's all on the CPU because I'm used to it and it looks nice

        void cce_loss(const float* outputs, const float* targets, float* results, const size_t n_batches, const size_t n_classes) {
            for (size_t batch = 0; batch < n_batches; batch++) {
                size_t c = 0;

                while (targets[c] != 1.0f && c < n_classes) c++;

                if (c == n_classes) std::runtime_error("No target for CCE loss computation");

                results[batch] = -logf(outputs[batch * n_classes + c]);
            }
        }

        void Context::computeLoss(const Matrix& outputs, const Matrix& targets, Matrix& results, const Loss loss) const {
            switch (loss) {
                case Loss::CCE:
                    cce_loss(outputs.data(), targets.data(), results.data(), outputs.rows(), outputs.columns());
                break;
            }
        }
    }
}