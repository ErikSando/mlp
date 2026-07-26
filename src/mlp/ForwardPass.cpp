#include <cassert>

#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::forwardPass(Batch& batch) {
        assert(m_batchSize * m_inputCount == batch.data.size());
        assert(m_layers.size());

        Matrix input(m_batchSize, m_inputCount);

        m_context.transfer(batch.data.data(), input);

        m_context.multiply(input, m_layers[0].weights, m_layers[0].nodes);
        m_context.addBiases(m_layers[0].nodes, m_layers[0].biases, m_layers[0].nodes);

        // activation function here

        // If I remember correctly, the output layer uses a seperate activation function. If so, change this:
        for (size_t layer = 1; layer < m_layers.size(); layer++) {
            m_context.multiply(m_layers[layer - 1].nodes, m_layers[layer].weights, m_layers[layer].nodes);
            m_context.addBiases(m_layers[layer].nodes, m_layers[layer].biases, m_layers[layer].nodes);

            // activation function here
        }
    }

    void MLP::copyOutputs(float* host_outputs) {
        m_context.transfer(m_layers.back().nodes, host_outputs);
    }
}