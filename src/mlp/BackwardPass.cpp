#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::backwardPass(const Batch& batch) {
        assert(m_batchSize == batch.size);
        assert(m_batchSize == batch.labels.size());

        Layer_up& output_layer = m_layers.back();
        Layer_up& last_hidden_layer = m_layers.at(m_layers.size() - 2);

        std::vector<Matrix> weight_gradients; // ordered backwards
        std::vector<Matrix> bias_gradients; // ordered backwards
        std::vector<Matrix> dC_da;

        weight_gradients.push_back(m_context.createMatrix(output_layer->weights.rows(), output_layer->weights.columns()));
        bias_gradients.push_back(m_context.createMatrix(1, output_layer->biases.size()));
        dC_da.push_back(m_context.createMatrix(m_batchSize, last_hidden_layer->activations.columns()));

        m_context.zeroMatrix(weight_gradients.back());
        m_context.zeroMatrix(bias_gradients.back());
        m_context.zeroMatrix(dC_da.back());

        m_context.computeOutputGradients(
            last_hidden_layer->activations, output_layer->activations,
            output_layer->weights,
            batch.labels,
            m_alPair,
            weight_gradients.back(), bias_gradients.back(), dC_da.back()
        );

        for (size_t l = m_layers.size() - 2; l > 0; l--) {
            Layer_up& layer = m_layers[l];
            Layer_up& preceding_layer = m_layers[l - 1];

            weight_gradients.push_back(m_context.createMatrix(layer->weights.rows(), layer->weights.columns()));
            bias_gradients.push_back(m_context.createMatrix(1, layer->biases.size()));
            dC_da.push_back(m_context.createMatrix(m_batchSize, preceding_layer->activations.columns()));

            m_context.zeroMatrix(weight_gradients.back());
            m_context.zeroMatrix(bias_gradients.back());
            m_context.zeroMatrix(dC_da.back());

            m_context.computeGradients(
                dC_da[dC_da.size() - 2],
                preceding_layer->activations, layer->activations,
                layer->weights,
                layer->activation,
                weight_gradients.back(), bias_gradients.back(), dC_da.back()
            );
        }

        for (int i = weight_gradients.size() - 1; i >= 0; i--) {
            Matrix& w_grads = weight_gradients.at(i);
            Matrix& b_grads = bias_gradients.at(i);
            Layer_up& layer = m_layers[m_layers.size() - 1 - i];

            m_context.optimiseLayer(layer->weights, layer->biases, w_grads, b_grads, m_learningRate);
        }
    }
}