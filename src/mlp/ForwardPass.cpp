#include <cassert>

#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::forwardPass(const Batch& batch) {
        assert(m_batchSize == m_layers[0].nodes.rows());
        assert(batch.data.size() == m_layers[0].nodes.size());

        m_context.transfer(batch.data.data(), m_layers[0].nodes);

        for (size_t layer = 1; layer < m_layers.size(); layer++) {
            m_layers[layer].propagate(m_context, m_layers[layer - 1].nodes);
        }
    }

    void MLP::copyOutputs(float* host_outputs) {
        m_context.transfer(m_layers.back().nodes, host_outputs);
    }
}