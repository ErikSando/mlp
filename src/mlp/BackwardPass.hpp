#pragma once

namespace mlp {
    template<typename TContext>
    void MLP<TContext>::backwardPass(const Batch& batch) {
        assert(m_batchSize == batch.size);
        assert(m_batchSize == batch.labels.size());

        using Layer_t_up = std::unique_ptr<Layer_t>;

        Layer_t_up& output_layer = m_layers.back();
        Layer_t_up& last_hidden_layer = m_layers.at(m_layers.size() - 2);

        std::vector<Matrix_t> gradients; // ordered backwards
        std::vector<Matrix_t> dC_da;

        gradients.emplace_back(output_layer->weights.rows(), output_layer->weights.columns()).zero();
        dC_da.emplace_back(m_batchSize, last_hidden_layer->activations.columns()).zero();

        m_context.computeOutputGradients(
            last_hidden_layer->activations, output_layer->activations,
            output_layer->weights,
            batch.labels,
            output_layer->activation, Loss::CCE,
            gradients.front(), dC_da.front()
        ); // loss set to CCE for now
    
        for (size_t l = m_layers.size() - 2; l > 0; l--) {
            Layer_t_up& layer = m_layers[l];
            Layer_t_up& preceding_layer = m_layers[l - 1];

            gradients.emplace_back(layer->weights.rows(), layer->weights.columns()).zero();
            dC_da.emplace_back(m_batchSize, preceding_layer->activations.columns()).zero();

            m_context.computeGradients(
                dC_da[dC_da.size() - 2],
                preceding_layer->activations, layer->activations,
                layer->weights,
                layer->activation,
                gradients.back(), dC_da.back()
            );

            float* host_gradients = new float[gradients.back().size()];
            m_context.transfer(host_gradients, gradients.back());

            delete[] host_gradients;
        }

        for (int i = gradients.size() - 1; i >= 0; i--) {
            Matrix_t& grads = gradients[i];
            Layer_t_up& layer = m_layers[m_layers.size() - 1 - i];

            m_context.optimiseLayer(layer->weights, grads, m_learningRate);
        }
    }
}