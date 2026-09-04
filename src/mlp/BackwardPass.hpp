#pragma once

namespace mlp {
    template<typename TContext>
    void MLP<TContext>::backwardPass(const Batch& batch) {
        assert(m_batchSize == batch.size);
        assert(m_batchSize == batch.labels.size());

        // using Layer_t_up = std::unique_ptr<Layer_t>;

        Layer_t_up& output_layer = m_layers.back();
        Layer_t_up& last_hidden_layer = m_layers.at(m_layers.size() - 2);

        std::vector<Matrix_t> weight_gradients; // ordered backwards
        std::vector<Matrix_t> bias_gradients; // ordered backwards
        std::vector<Matrix_t> dC_da;

        weight_gradients.emplace_back(output_layer->weights.rows(), output_layer->weights.columns()).zero();
        bias_gradients.emplace_back(1, output_layer->biases.size()).zero();
        dC_da.emplace_back(m_batchSize, last_hidden_layer->activations.columns()).zero();

        m_context.computeOutputGradients(
            last_hidden_layer->activations, output_layer->activations,
            output_layer->weights,
            batch.labels,
            // output_layer->activation, m_lossFunction,
            m_alPair,
            weight_gradients.back(), bias_gradients.back(), dC_da.back()
        );

        // std::vector<float> host_bias_gradients(bias_gradients.back().size());

        // m_context.transfer(host_bias_gradients.data(), bias_gradients.back());

        // std::cout << "Output bias gradients\n\n";
        // for (float gradient : host_bias_gradients) {
        //     std::cout << gradient << " ";
        // }
        // std::cout << "\n\n";

        for (size_t l = m_layers.size() - 2; l > 0; l--) {
            Layer_t_up& layer = m_layers[l];
            Layer_t_up& preceding_layer = m_layers[l - 1];

            weight_gradients.emplace_back(layer->weights.rows(), layer->weights.columns()).zero();
            bias_gradients.emplace_back(1, layer->biases.size()).zero();
            dC_da.emplace_back(m_batchSize, preceding_layer->activations.columns()).zero();

            m_context.computeGradients(
                dC_da[dC_da.size() - 2],
                preceding_layer->activations, layer->activations,
                layer->weights,
                layer->activation,
                weight_gradients.back(), bias_gradients.back(), dC_da.back()
            );

            // std::vector<float> host_bias_gradients(bias_gradients.back().size());

            // m_context.transfer(host_bias_gradients.data(), bias_gradients.back());

            // std::cout << "Hidden layer bias gradients\n\n";
            // for (float gradient : host_bias_gradients) {
            //     std::cout << gradient << " ";
            // }
            // std::cout << "\n\n";
        }

        for (int i = weight_gradients.size() - 1; i >= 0; i--) {
            Matrix_t& w_grads = weight_gradients.at(i);
            Matrix_t& b_grads = bias_gradients.at(i);
            Layer_t_up& layer = m_layers[m_layers.size() - 1 - i];

            m_context.optimiseLayer(layer->weights, layer->biases, w_grads, b_grads, m_learningRate);
        }
    }
}