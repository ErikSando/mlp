#include "opencl/Context.hpp"

namespace mlp {
    namespace opencl {
        void Context::computeGradients(
            const Matrix_t& dC_da,
            const Matrix_t& left_activations, const Matrix_t& right_activations,
            const Matrix_t& weights,
            const Activation activation,
            Matrix_t& weight_gradients, Matrix_t& bias_gradients, Matrix_t& dC_da_next
        ) const {
            
        }

        void Context::computeOutputGradients(
            const Matrix_t& last_activations, const Matrix_t& output_activations,
            const Matrix_t& weights,
            const std::vector<int>& labels,
            const OALP al_pair,
            Matrix_t& weight_gradients, Matrix_t& bias_gradients, Matrix_t& dC_da_hidden
        ) const {
            
        }

        void Context::optimiseLayer(Matrix_t& weights, Matrix& biases, const Matrix_t& weight_gradients, const Matrix_t& bias_gradients, const float learning_rate) const {

        }
    }
}