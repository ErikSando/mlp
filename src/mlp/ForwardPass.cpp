#include <cassert>

#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::forwardPass(Batch& batch) {
        assert(m_batchSize == m_layers[0].getNodes().rows());
        assert(batch.data.size() == m_layers[0].getNodes().size());

        m_layers[0].uploadNodes(m_context, batch.data.data());

        for (size_t layer = 1; layer < m_layers.size(); layer++) {
            m_layers[layer].propagate(m_context, m_layers[layer - 1].getNodes());
        }

        m_context.synchronise();
    }

    void MLP::copyOutputs(float* host_outputs) {
        m_context.transfer(m_layers.back().getNodes(), host_outputs);
    }
}