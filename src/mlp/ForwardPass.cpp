#include <cassert>

#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::forwardPass(const Batch& batch) {
        assert(m_batchSize == m_layers[0]->activations.rows());
        assert(batch.data.size() == m_layers[0]->activations.size());

        m_context.transfer(m_layers[0]->activations, batch.data.data());

        for (size_t l = 1; l < m_layers.size(); l++) {
            // m_layers[l]->propagate(m_context, m_layers[l - 1]->activations);

            m_context.propagate(
                m_layers[l - 1]->activations, m_layers[l]->logits, m_layers[l]->activations,
                m_layers[l]->weights, m_layers[l]->biases,
                m_layers[l]->activation
            );
        }
    }

    void MLP::copyOutputs(float* host_outputs) const {
        m_context.transfer(host_outputs, m_layers.back()->activations);
    }
}