#include "device/DeviceContext.hpp"
#include "mlp/Layer.hpp"

namespace mlp {
    void Layer::uploadNodes(const DeviceContext& context, const float* new_nodes) {
        context.transfer(new_nodes, m_nodes);
    }

    void Layer::propagate(const DeviceContext& context, const Matrix& previous_layer) {
        context.multiply(previous_layer, m_weights, m_nodes);
        context.addBiases(m_nodes, m_biases, m_nodes);

        switch (m_activation) {
            case Activation::SIGMOID:      context.sigmoid(m_nodes, m_nodes); break;
            case Activation::TANH:         context.tanh(m_nodes, m_nodes); break;
            case Activation::RELU:         context.relu(m_nodes, m_nodes); break;
            case Activation::LEAKY_RELU:   context.leakyReLU(m_nodes, m_nodes); break;

            case Activation::SOFTMAX:      context.softmax(m_nodes, m_nodes); break;

            default: break;
        }
    }
}