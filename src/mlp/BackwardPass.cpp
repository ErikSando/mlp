#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::backwardPass(const Batch& batch) {
        Layer& output_layer = m_layers.back();
        Layer& last_hidden_layer = m_layers.at(m_layers.size() - 2);

        Matrix gradients(output_layer.weights.rows(), output_layer.weights.columns());
        Matrix dC_da(m_batchSize, last_hidden_layer.nodes.columns());

        m_context.computeOutputGradients(
            last_hidden_layer.nodes, output_layer.nodes, output_layer.weights,
            last_hidden_layer.nodes.columns(),
            batch.labels,
            output_layer.activation, Loss::CCE,
            gradients, dC_da
        ); // loss set to CCE for now

        m_context.optimiseLayer(output_layer.weights, gradients, m_learningRate);

        for (size_t layer = m_layers.size() - 2; layer > 0; layer--) {
            
        }

        float* host_gradients = new float[gradients.size()];
        m_context.transfer(gradients, host_gradients);

        for (size_t i = 0; i < gradients.size(); i++) {
            std::cout << "  " << host_gradients[i];
        }

        std::cout << "\n";

        float* h_dC_da = new float[dC_da.size()];
        m_context.transfer(dC_da, h_dC_da);

        for (size_t i = 0; i < dC_da.size(); i++) {
            std::cout << "  " << h_dC_da[i];
        }

        std::cout << "\n";
    }
}