#include "device/DeviceContext.hpp"
#include "mlp/Layer.hpp"

namespace mlp {
    void Layer::uploadNodes(const DeviceContext& context, const float* new_nodes) {
        context.transfer(new_nodes, m_nodes);
    }

    void Layer::propagate(const DeviceContext& context, const Matrix& previous_layer) {
        context.propagate(previous_layer, m_nodes, m_weights, m_biases, m_activation);
    }

    void Layer::propagateOld(const DeviceContext& context, const Matrix& previous_layer) {
        context.propagateOld(previous_layer, m_nodes, m_weights, m_biases, m_activation);
    }
}