#include "mlp/MLP.hpp"

namespace mlp {
    template<typename TContext>
    void MLP<TContext>::checkOutputs(const std::vector<int>& labels, Buffer_t& correct, Buffer_t& classifications) const {
        m_context.checkOutputs(m_layers.back()->activations, labels, correct, classifications);
    }
}