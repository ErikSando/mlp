#include <cmath>
#include <iostream>

#include "mlp/MLP.hpp"

namespace mlp {
    MLP::MLP(DeviceContext& device_context, std::vector<size_t>& layer_sizes, const size_t batch_size)
    : m_context(device_context), m_batchSize(batch_size)
    {
        m_inputCount = layer_sizes[0];
        size_t previous_count = m_inputCount;

        float weight_max = std::sqrt(2.0f / static_cast<float>(m_inputCount));
        float weight_min = -weight_max;

        for (size_t i = 1; i < layer_sizes.size(); i++) {
            size_t node_count = layer_sizes[i];

            m_layers.emplace_back(batch_size, node_count, previous_count);

            previous_count = node_count;

            m_context.randomise(m_layers.back().weights, weight_min, weight_max);
        }
    }

    MLP::~MLP() {

    }
}