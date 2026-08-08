#include "device/DeviceContext.hpp"
#include "matrix/HostMatrix.hpp"

namespace mlp {
    class HostContext : public IDeviceContext<HostMatrix> {
        public:

        using Matrix = HostMatrix;
 
        void transfer(const Matrix& src, float* dest) const override;
        void transfer(const float* src, Matrix& dest) const override;

        // Randomise each value in the matrix to a value between min and max
        void randomise(Matrix& matrix, float min, float max) const override;

        void propagate(
            const Matrix& last_activations,
            Matrix& logits, Matrix& activations,
            const Matrix& weights, const Matrix& biases,
            const Activation activation
        ) const override;

        void computeOutputGradients(
            const Matrix& last_activations, const Matrix& activations, const Matrix& weights,
            const size_t n_last_activations,
            const std::vector<int>& labels,
            const Activation activation, const Loss loss,
            Matrix& gradients, Matrix& dC_da_next
        ) const override; // output layer

        void computeGradients(
            const Matrix& dC_da,
            const Matrix& left_activations, const Matrix& right_activations,
            const Matrix& weights,
            const Activation activation,
            Matrix& gradients, Matrix& dC_da_next
        ) const override; // hidden layers

        void optimiseLayer(Matrix& weights, const Matrix& gradients, const float learning_rate) const override;

        void checkOutputs(const Matrix& outputs, const std::vector<int>& labels, const size_t n_samples, Matrix& correct, Matrix& classifications) const override;
    };
}