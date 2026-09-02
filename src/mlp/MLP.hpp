#pragma once

#include <memory>

#include "data/Batch.hpp"
#include "enums/Enums.hpp"
#include "mlp/Layer.hpp"

namespace mlp {
    constexpr float UNDEFINED_CLASS = -1;
    constexpr float UNDEFINED_ERROR = -1.0f;

    struct ClassifyInfo {
        int classification = UNDEFINED_CLASS;
        float error = UNDEFINED_ERROR;
        std::vector<float> outputs;
    };

    struct ModelData {
        std::vector<size_t> layer_sizes;
        Activation hidden_activation = Activation::NONE;
        Activation output_activation = Activation::NONE;
        Loss loss_function;
        size_t batch_size;
        float learning_rate;
        std::vector<std::vector<float>> weights;
        std::vector<std::vector<float>> biases;
    };

    template<typename TContext>
    class MLP {
        public:

        using Buffer_t = typename TContext::Buffer_t;
        using Matrix_t = typename TContext::Matrix_t;
        using Layer_t = Layer<TContext>;
        using Layer_t_up = std::unique_ptr<Layer_t>;

        MLP(TContext& device_context) : m_context(device_context) {}
        ~MLP() {}

        // Construct the layers of the network, specifying the sizes and activation functions
        void init(
            std::vector<size_t>& layer_sizes,
            const size_t batch_size = 32UL,
            const Activation hidden_activation = Activation::LEAKY_RELU,
            const Activation output_activation = Activation::SOFTMAX,
            const Loss loss_function = Loss::CCE,
            const float learning_rate = 0.01f,
            const bool init_weights = true
        );

        void init(const ModelData& data);

        const TContext& getContext() const { return m_context; }

        void classify(const Sample& sample, ClassifyInfo& info);

        void forwardPass(const Batch& batch);
        void backwardPass(const Batch& batch);

        // Check the number of correct classifications out of the given number of samples
        // I removed giving a number of samples, I don't know why I had that in the first place, maybe to look at the first n samples in a batch?
        // I'll add it back later if I need it
        // Just realised, I need to be able to ignore later samples in case the total number of samples in the dataset is not a multiple of the batch size
        void checkOutputs(const std::vector<int>& labels, Buffer_t& correct, Buffer_t& classifications) const;

        // Copy the values of the output nodes into the given host memory location
        void copyOutputs(float* host_outputs) const;
        // ^^^ use a safer method in the future, the size of host_outputs isnt enforced right now

        size_t getInputCount() const { return m_layers[0]->logits.columns(); }
        constexpr size_t getBatchSize() const { return m_batchSize; }

        float getLearningRate() const { return m_learningRate; }
        void setLearningRate(float learning_rate) { m_learningRate = learning_rate; }

        // I've decided I'm going to stick to as few getters as possible and just provide a function to export the model's data

        // not sure what a good API would be, so I'm just giving a const reference to the layers vector
        // const std::vector<Layer_t_up>& getLayers() const { return m_layers; }
        // I don't think I need a non-const reference, because no values in the layer are changed (cos it uses pointers)

        // const Loss getLossFunction() const { return m_lossFunction; }

        void exportData(ModelData& data);

        private:

        std::vector<Layer_t_up> m_layers;

        TContext& m_context;
        size_t m_batchSize;
        float m_learningRate;
        Loss m_lossFunction;
    };
}

#include "mlp/BackwardPass.hpp"
#include "mlp/CheckOutputs.hpp"
#include "mlp/Classify.hpp"
#include "mlp/ExportData.hpp"
#include "mlp/ForwardPass.hpp"
#include "mlp/Init.hpp"