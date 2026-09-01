#pragma once

#include <cmath>
#include <iostream>

#include <random>

namespace mlp {
    template<typename TContext>
    void MLP<TContext>::init(
        std::vector<size_t>& layer_sizes,
        const size_t batch_size,
        const Activation hidden_activation, const Activation output_activation,
        const Loss loss_function,
        const float learning_rate
    ) {
        m_batchSize = batch_size;
        m_lossFunction = loss_function;
        m_learningRate = learning_rate;

        size_t node_count = layer_sizes[0];

        m_layers.emplace_back(std::make_unique<Layer_t>(m_batchSize, node_count, 0));

        size_t previous_count = node_count;

        // using an equal seed to test that different builds give the same results
        constexpr unsigned int SEED = 1234567890;
        std::mt19937 gen(SEED);

        for (size_t i = 1; i < layer_sizes.size(); i++) {
            size_t node_count = layer_sizes[i];

            float weight_max = std::sqrt(2.0f / static_cast<float>(node_count));
            float weight_min = -weight_max;

            std::uniform_real_distribution<float> distrib(weight_min, weight_max);

            bool is_output_layer = i == layer_sizes.size() - 1;
            Activation activation = is_output_layer ? output_activation : hidden_activation;

            m_layers.push_back(std::make_unique<Layer_t>(m_batchSize, node_count, previous_count, activation));
            // m_context.randomise(m_layers.back()->weights, weight_min, weight_max);
            m_layers.back()->biases.zero();

            float* weights = new float[m_layers.back()->weights.size()];
            for (size_t i = 0; i < m_layers.back()->weights.size(); i++) {
                weights[i] = distrib(gen);
            }
            m_context.transfer(m_layers.back()->weights, weights);

            previous_count = node_count;
        }
    }
}