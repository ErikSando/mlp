#include "mlp/MLP.hpp"

namespace mlp {
    template<typename TDeviceContext>
    void MLP<TDeviceContext>::checkOutputs(const std::vector<int>& labels, const size_t n_samples, Matrix& correct, Matrix& classifications) {
        m_context.checkOutputs(m_layers.back().activations, labels, n_samples, correct, classifications);
    }
}