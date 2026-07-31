#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::backwardPass(const Batch& batch) {
        Layer& output_layer = m_layers.back();
        Layer& last_hidden_layer = m_layers.at(m_layers.size() - 2);

        Matrix gradients(output_layer.weights.rows(), output_layer.weights.columns());

        m_context.computeGradients(last_hidden_layer.nodes, output_layer.nodes, last_hidden_layer.nodes.columns(), batch.labels, output_layer.activation, Loss::CCE, gradients); // loss set to CCE for now

        float* host_gradients = new float[gradients.size()];
        m_context.transfer(gradients, host_gradients);

        for (size_t i = 0; i < gradients.size(); i++) {
            std::cout << "  " << host_gradients[i];
        }

        std::cout << "\n";
    }
}