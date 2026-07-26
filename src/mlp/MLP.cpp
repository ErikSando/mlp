#include <cmath>
#include <iostream>

#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::init(std::vector<size_t>& layer_sizes, const Activation hidden_activation, const Activation output_activation) {
        size_t node_count = layer_sizes[0];

        float weight_max = std::sqrt(2.0f / static_cast<float>(node_count));
        float weight_min = -weight_max;

        m_layers.emplace_back(m_batchSize, node_count, 0);

        size_t previous_count = node_count;

        for (size_t i = 1; i < layer_sizes.size(); i++) {
            size_t node_count = layer_sizes[i];

            m_layers.emplace_back(m_batchSize, node_count, previous_count, i == layer_sizes.size() - 1 ? output_activation : hidden_activation);

            previous_count = node_count;

            m_context.randomise(m_layers.back().weights, weight_min, weight_max);
        }
    }

    MLP::~MLP() {

    }
}