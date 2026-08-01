#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::checkOutputs(const std::vector<int>& labels, const size_t n_samples, Matrix& correct, Matrix& classifications) {
        m_context.checkOutputs(m_layers.back().nodes, labels, n_samples, correct, classifications);
    }
}