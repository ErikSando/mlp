#pragma once

#include "data/Batch.hpp"
#include "device/DeviceContext.hpp"
// #include "matrix/Matrix.hpp"
#include "mlp/Layer.hpp"

// things are very experimental right now

namespace mlp {
    class MLP {
        public:

        MLP(DeviceContext& device_context, const size_t batch_size = 32, const float learning_rate = 0.01f)
        : m_context(device_context), m_batchSize(batch_size), m_loss(batch_size, 1), m_learningRate(learning_rate) {}

        ~MLP();

        // Construct the layers of the network, specifying the sizes and activation functions
        void init(
            std::vector<size_t>& layer_sizes,
            const Activation hidden_activation = Activation::LEAKY_RELU,
            const Activation output_activation = Activation::SOFTMAX,
            const Loss loss_function = Loss::CCE
        );

        void forwardPass(const Batch& batch);
        void backwardPass(const Batch& batch);

        // im thinking the forward pass function will have no return type and pass the values through the network
        // and the output values can be read using another function
        // i think this makes sense considering the node values are on the GPU anyway

        // Copy the values of the output nodes into the given host memory location
        void copyOutputs(float* host_outputs);
        // ^^^ use a safer method in the future, the size of host_outputs isnt enforced right now

        // Check the number of correct classifications out of the given number of samples
        void checkOutputs(const std::vector<int>& labels, const size_t n_samples, Matrix& correct, Matrix& classifications);

        constexpr size_t getInputCount() { return m_layers[0].nodes.columns(); }
        constexpr size_t getBatchSize() { return m_batchSize; }

        private:

        Batch* last_batch = nullptr;

        DeviceContext& m_context;

        size_t m_batchSize;
        
        float m_learningRate;

        std::vector<Layer> m_layers; // includes all layers: input, hidden, output

        Loss m_lossFunction;
        Matrix m_loss; // used as a vector
    };
}