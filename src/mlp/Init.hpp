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
        const float learning_rate,
        const bool init_weights
    ) {
        m_batchSize = batch_size;
        m_lossFunction = loss_function;
        m_learningRate = learning_rate;

        if (layer_sizes.size() < 2) {
            throw std::runtime_error("At least 2 layers are required");
            return;
        }

        size_t node_count = layer_sizes[0];

        m_layers.emplace_back(std::make_unique<Layer_t>(m_batchSize, node_count, 0));

        size_t previous_count = node_count;

        // using an equal seed to test that different builds give the same results
        constexpr unsigned int SEED = 12345;
        std::mt19937 gen(SEED);

        for (size_t i = 1; i < layer_sizes.size(); i++) {
            size_t node_count = layer_sizes[i];

            bool is_output_layer = i == layer_sizes.size() - 1;
            Activation activation = is_output_layer ? output_activation : hidden_activation;

            m_layers.push_back(std::make_unique<Layer_t>(m_batchSize, node_count, previous_count, activation));
            m_layers.back()->biases.zero();

            if (init_weights) {
                // m_context.randomise(m_layers.back()->weights, weight_min, weight_max);

                float weight_max = std::sqrt(2.0f / static_cast<float>(node_count));
                float weight_min = -weight_max;

                std::uniform_real_distribution<float> distrib(weight_min, weight_max);

                float* weights = new float[m_layers.back()->weights.size()];
                for (size_t i = 0; i < m_layers.back()->weights.size(); i++) {
                    weights[i] = distrib(gen);
                }
                m_context.transfer(m_layers.back()->weights, weights);
            }

            previous_count = node_count;
        }
    }

    template<typename TContext>
    void MLP<TContext>::init(const ModelData& data) {
        if (data.weights.size() != data.layer_sizes.size() - 1) {
            // throw std::runtime_error("Incorrect number of weight matrices given, expected " + data.layer_sizes.size() - 1 + ", got " + data.weights.size());
            throw std::runtime_error("Incorrect number of weight matrices given");
        }

        if (data.biases.size() != data.layer_sizes.size() - 1) {
            // throw std::runtime_error("Incorrect number of bias vectors given, expected " + data.layer_sizes.size() - 1 + ", got " + data.biases.size());
            throw std::runtime_error("Incorrect number of bias vectors given");
        }

        init(data.layer_sizes, data.batch_size, data.hidden_activation, data.output_activation, data.loss_function, data.learning_rate, false);

        for (size_t i = 1; i < m_layers.size(); i++) {
            Layer_t_up& layer = m_layers.at(i);
            std::vector<float>& weights = data.weights.at(i - 1);
            std::vector<float>& biases = data.biases.at(i - 1);

            if (weights.size() != layer->weights.size()) {
                // throw std::runtime_error("Incorrect number of weights given, expected " + layer->weights.size() + ", got " + weights.size());
                throw std::runtime_error("Incorrect number of weights given");
            }

            if (biases.size() != layer->biases.size()) {
                // throw std::runtime_error("Incorrect number of biases given, expected " + layer->biases.size() + ", got " + biases.size());
                throw std::runtime_error("Incorrect number of biases given");
            }

            m_context.transfer(layer->weights, weights.data());
            m_context.transfer(layer->biases, biases.data());
        }
    }
}