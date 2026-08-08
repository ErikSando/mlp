#pragma once

#include "data/Batch.hpp"
#include "enums/Enums.hpp"
#include "mlp/Layer.hpp"

// things are very experimental right now

namespace mlp {
    constexpr float UNDEFINED_CLASS = -1;
    constexpr float UNDEFINED_ERROR = -1.0f;

    struct ClassifyInfo {
        int classification = UNDEFINED_CLASS;
        float error = UNDEFINED_ERROR;
    };

    template<typename TContext>
    class MLP {
        using Matrix = typename TContext::Matrix;

        public:

        MLP(TContext& device_context) : m_context(device_context) {}
        ~MLP() {}

        // Construct the layers of the network, specifying the sizes and activation functions
        void init(
            std::vector<size_t>& layer_sizes,
            const size_t batch_size = 32UL,
            const Activation hidden_activation = Activation::LEAKY_RELU,
            const Activation output_activation = Activation::SOFTMAX,
            const Loss loss_function = Loss::CCE,
            const float learning_rate = 0.01f
        );

        void classify(const Sample& sample, ClassifyInfo& info);

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

        size_t getInputCount() { return m_layers[0].logits.columns(); }
        constexpr size_t getBatchSize() { return m_batchSize; }

        float getLearningRate() { return m_learningRate; }
        void setLearningRate(float learning_rate) { m_learningRate = learning_rate; }

        private:

        std::vector<Layer<TContext>> m_layers; // includes all layers: input, hidden, output

        TContext& m_context;
        size_t m_batchSize;
        float m_learningRate;
        Loss m_lossFunction;
    };
}

#include "mlp/BackwardPass.hpp"
#include "mlp/CheckOutputs.hpp"
#include "mlp/Classify.hpp"
#include "mlp/ForwardPass.hpp"
#include "mlp/Init.hpp"