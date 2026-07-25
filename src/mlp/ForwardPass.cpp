#include <cassert>

#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::forwardPass(Batch& batch) {
        assert(m_batchSize * m_inputCount == batch.data.size());

        Matrix input(m_batchSize, m_inputCount);

        m_context.transfer(batch.data.data(), input);

        assert(
            m_weights.size() && m_layers.size() && m_biases.size() &&
            m_weights.size() == m_layers.size() && m_layers.size() == m_biases.size()
        );

        m_context.multiply(input, m_weights[0], m_layers[0]);
        m_context.addBiases(m_layers[0], m_biases[0], m_layers[0]);

        // activation function here

        // If I remember correctly, the output layer uses a seperate activation function. If so, change this:
        for (size_t layer = 1; layer < m_layers.size(); layer++) {
            m_context.multiply(m_layers[layer - 1], m_weights[layer], m_layers[layer]);
            m_context.addBiases(m_layers[layer], m_biases[layer], m_layers[layer]);

            // activation function here
        }
    }

    void MLP::copyOutputs(float* host_outputs) {
        m_context.transfer(m_layers.back(), host_outputs);
    }
}