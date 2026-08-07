#include <cmath>
#include <iostream>

#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::init(std::vector<size_t>& layer_sizes, const Activation hidden_activation, const Activation output_activation, const Loss loss_function)
    {
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

            float weights[16] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 };
            // float weights[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            m_layers.emplace_back(m_batchSize, node_count, previous_count, activation);
            // m_context.randomise(m_layers.back().weights, weight_min, weight_max);

            m_context.transfer(weights, m_layers.back().weights);

            previous_count = node_count;
        }
    }

    MLP::~MLP() {

    }
}