#include "opencl/Context.hpp"

namespace mlp {
    namespace opencl {
        void Context::computeGradients(
            const Matrix& dC_da,
            const Matrix& left_activations, const Matrix& right_activations,
            const Matrix& weights,
            const Activation activation,
            Matrix& weight_gradients, Matrix& bias_gradients, Matrix& dC_da_next
        ) const {
            
        }

        void Context::computeOutputGradients(
            const Matrix& last_activations, const Matrix& output_activations,
            const Matrix& weights,
            const std::vector<int>& labels,
            const OALP al_pair,
            Matrix& weight_gradients, Matrix& bias_gradients, Matrix& dC_da_hidden
        ) const {
            
        }

        void Context::optimiseLayer(Matrix& weights, Matrix& biases, const Matrix& weight_gradients, const Matrix& bias_gradients, const float learning_rate) const {

        }
    }
}