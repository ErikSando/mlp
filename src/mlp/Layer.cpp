#include "device/DeviceContext.hpp"
#include "mlp/Layer.hpp"

namespace mlp {
    void Layer::propagate(const DeviceContext& context, const Matrix& previous_layer) {
        context.propagate(previous_layer, logits, activations, weights, biases, activation);
    }
}