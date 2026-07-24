#include <cmath>
#include <iostream>

#include "mlp/MLP.hpp"

namespace mlp {
    MLP::MLP(DeviceContext& device_context, std::vector<size_t>& layer_sizes) : m_context(device_context) {
        size_t input_count = layer_sizes[0];
        size_t previous_count = input_count;

        float weight_max = std::sqrt(2.0f / static_cast<float>(input_count));
        float weight_min = -weight_max;

        std::cout << "Weights: ±" << weight_max << "\n";

        for (size_t i = 1; i < layer_sizes.size(); i++) {
            size_t node_count = layer_sizes[i];
            size_t rows = previous_count;

            previous_count = node_count;

            m_layers.emplace_back(1, node_count); // replace the 1 with the batch size later
            m_weights.emplace_back(rows, node_count);

            m_context.randomise(m_weights.back(), weight_min, weight_max);
        }

        // check sizes

        for (int i = 0; i < m_weights.size(); i++) {
            Matrix& weights = m_weights[i];
            std::cout << "Weights " << i << " size: " << weights.size() << " (" << weights.rows() << "x" << weights.columns() << ")\n";
        }

        for (int i = 0; i < m_layers.size(); i++) {
            Matrix& nodes = m_layers[i];
            std::cout << "Layer " << i << " size: " << nodes.size() << " (" << nodes.rows() << "x" << nodes.columns() << ")\n";
        }

        // check values

        for (int i = 0; i < m_weights.size(); i++) {
            Matrix& weights = m_weights[i];
            float* values = new float[weights.size()];

            m_context.transfer(weights, values);

            std::cout << "Weights " << i << " values:\n";

            std::cout << values[0];

            for (size_t i = 1; i < weights.size(); i++) {
                std::cout << ", " << values[i];
            }

            std::cout << "\n";

            delete[] values;
        }

        for (int i = 0; i < m_layers.size(); i++) {
            Matrix& nodes = m_layers[i];
            float* values = new float[nodes.size()];

            m_context.transfer(nodes, values);

            std::cout << "Layer " << i << " values:\n";

            std::cout << values[0];

            for (size_t i = 1; i < nodes.size(); i++) {
                std::cout << ", " << values[i];
            }

            std::cout << "\n";

            delete[] values;
        }
    }

    MLP::~MLP() {

    }
}