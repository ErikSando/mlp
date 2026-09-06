#include "opencl/Context.hpp"

namespace mlp {
    namespace opencl {
        void Context::propagate(
            const Matrix& last_activations,
            Matrix& logits, Matrix& activations,
            const Matrix& weights, const Matrix& biases,
            const Activation activation
        ) const {
            
        }
    }
}