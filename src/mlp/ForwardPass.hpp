#pragma once

#include <cassert>

namespace mlp {
    template<typename TContext>
    void MLP<TContext>::forwardPass(const Batch& batch) {
        assert(m_batchSize == m_layers[0]->activations.rows());
        assert(batch.data.size() == m_layers[0]->activations.size());

        m_context.transfer(batch.data.data(), m_layers[0]->activations);

        for (size_t layer = 1; layer < m_layers.size(); layer++) {
            m_layers[layer]->propagate(m_context, m_layers[layer - 1]->activations);
        }
    }

    template<typename TContext>
    void MLP<TContext>::copyOutputs(float* host_outputs) {
        m_context.transfer(m_layers.back()->activations, host_outputs);
    }
}