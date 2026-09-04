#include "host/Context.hpp"

namespace mlp {
    namespace host {
        void check_outputs(const float* outputs, const int* labels, const size_t rows, const size_t cols, int* correct, int* classifications) {
            for (size_t row = 0; row < rows; row++) {
                int classification = -1;
                float max = 0.0f;

                for (int col = 0; col < cols; col++) {
                    float output = outputs[row * cols + col];

                    if (output > max) {
                        classification = col;
                        max = output;
                    }
                }

                if (classification == labels[row]) correct[0]++;

                classifications[row] = classification;
            }
        }

        void Context::checkOutputs(const Matrix_t& outputs, const std::vector<int>& labels, Buffer_t& correct, Buffer_t& classifications) const {
            assert(outputs.rows() == labels.size());

            // correct.zero();
            // classifications.zero();

            if (m_profiler) m_profiler->startTask("Check Outputs");

            check_outputs(outputs.data(), labels.data(), outputs.rows(), outputs.columns(), (int*) correct.data(), (int*) classifications.data());

            if (m_profiler) m_profiler->endTask("Check Outputs");
        }
    }
}