#include "device/HostContext.hpp"

namespace mlp {
    void HostContext::propagate(
        const Matrix& inputs,
        Matrix& logits, Matrix& activations,
        const Matrix& weights, const Matrix& biases,
        const Activation activation
    ) const {
        
    }
}