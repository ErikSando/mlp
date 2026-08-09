#include "mlp/MLP.hpp"

namespace mlp {
    template<typename TContext>
    void MLP<TContext>::checkOutputs(const std::vector<int>& labels, const size_t n_samples, Matrix_t& correct, Matrix_t& classifications) {
        m_context.checkOutputs(m_layers.back()->activations, labels, n_samples, correct, classifications);
    }
}