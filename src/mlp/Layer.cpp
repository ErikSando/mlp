#include "device/DeviceContext.hpp"
#include "mlp/Layer.hpp"

namespace mlp {
    void Layer::propagate(const DeviceContext& context, const Matrix& previous_layer) {
        context.multiply(previous_layer, weights, nodes);
        context.addBiases(nodes, biases, nodes);

        switch (activation) {
            case Activation::SIGMOID:      context.sigmoid(nodes, nodes); break;
            case Activation::TANH:         context.tanh(nodes, nodes); break;
            case Activation::RELU:         context.relu(nodes, nodes); break;
            case Activation::LEAKY_RELU:   context.leakyReLU(nodes, nodes); break;

            case Activation::SOFTMAX:      context.softmax(nodes, nodes); break;

            default: break;
        }
    }
}