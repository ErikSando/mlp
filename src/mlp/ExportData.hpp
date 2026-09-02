#pragma once

namespace mlp {
    template<typename TContext>
    void MLP<TContext>::exportData(ModelData& data) {
        data.layer_sizes.clear();

        for (auto& layer : m_layers) {
            data.layer_sizes.push_back(layer->activations.columns());
        }

        Activation hidden_activation = Activation::NONE;

        if (m_layers.size() >= 3) {
            hidden_activation = m_layers[1]->activation;
        }

        data.hidden_activation = hidden_activation;
        data.output_activation = m_layers.back()->activation;
        data.loss_function = m_lossFunction;
        data.batch_size = m_batchSize;
        data.learning_rate = m_learningRate;

        data.weights.clear();
        data.biases.clear();

        for (size_t i = 1; i < m_layers.size(); i++) {
            Layer_t_up& layer = m_layers.at(i);

            data.weights.emplace_back(layer->weights.size());
            data.biases.emplace_back(layer->biases.size());

            m_context.transfer(data.weights.back().data(), layer->weights);
            m_context.transfer(data.biases.back().data(), layer->biases);
        }
    }
}