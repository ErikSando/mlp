#include "opencl/Context.hpp"

namespace mlp {
    namespace opencl {
        void Context::propagate(
            const Matrix_t& last_activations,
            Matrix_t& logits, Matrix_t& activations,
            const Matrix_t& weights, const Matrix_t& biases,
            const Activation activation
        ) const {
            
        }
    }
}