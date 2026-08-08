#pragma once

#include <cmath>
#include <iostream>

namespace mlp {
    template<typename TDeviceContext>
    void MLP<TDeviceContext>::init(std::vector<size_t>& layer_sizes, const Activation hidden_activation, const Activation output_activation, const Loss loss_function) {
        m_lossFunction = loss_function;

        size_t node_count = layer_sizes[0];

        float weight_max = std::sqrt(2.0f / static_cast<float>(node_count));
        float weight_min = -weight_max;

        m_layers.emplace_back(m_batchSize, node_count, 0);

        size_t previous_count = node_count;

        for (size_t i = 1; i < layer_sizes.size(); i++) {
            size_t node_count = layer_sizes[i];

            bool isOutputLayer = i == layer_sizes.size() - 1;
            Activation activation = isOutputLayer ? output_activation : hidden_activation;

            m_layers.emplace_back(m_batchSize, node_count, previous_count, activation);
            m_context.randomise(m_layers.back().weights, weight_min, weight_max);

            m_layers.back().biases.zero();

            previous_count = node_count;
        }
    }
}