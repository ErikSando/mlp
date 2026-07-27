#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    struct Layer {
        Layer(const size_t batch_size, const size_t node_count, const size_t previous_count, const Activation activation = Activation::LEAKY_RELU)
        : m_nodes(batch_size, node_count), m_weights(previous_count, node_count), m_biases(1, node_count), m_activation(activation) {}

        void uploadNodes(const DeviceContext& context, const float* new_nodes);
        Matrix& getNodes() { return m_nodes; }
        Matrix& getWeights() { return m_weights; }

        void propagate(const DeviceContext& context, const Matrix& previous_layer);

        void propagateOld(const DeviceContext& context, const Matrix& previous_layer); // for testing

        private:

        Matrix m_nodes;
        Matrix m_weights;
        Matrix m_biases;

        Activation m_activation;
    };
}