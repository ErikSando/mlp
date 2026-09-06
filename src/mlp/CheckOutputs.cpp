#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::checkOutputs(const std::vector<int>& labels, Buffer& correct, Buffer& classifications, const size_t samples) const {
        m_context.checkOutputs(m_layers.back()->activations, labels, correct, classifications, samples > 0 ? samples : m_layers.back()->activations.rows());
    }
}