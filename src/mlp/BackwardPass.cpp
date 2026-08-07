#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::backwardPass(const Batch& batch) {
        Layer& output_layer = m_layers.back();
        Layer& last_hidden_layer = m_layers.at(m_layers.size() - 2);

        Matrix gradients(output_layer.weights.rows(), output_layer.weights.columns());
        Matrix dC_da(m_batchSize, last_hidden_layer.activations.columns());

        m_context.computeOutputGradients(
            last_hidden_layer.activations, output_layer.activations, output_layer.weights,
            last_hidden_layer.activations.columns(),
            batch.labels,
            output_layer.activation, Loss::CCE,
            gradients, dC_da
        ); // loss set to CCE for now

        // m_context.optimiseLayer(output_layer.weights, gradients, m_learningRate);

        for (size_t layer = m_layers.size() - 2; layer > 0; layer--) {
            
        }

        float* host_gradients = new float[gradients.size()];
        m_context.transfer(gradients, host_gradients);

        for (size_t i = 0; i < gradients.size(); i++) {
            std::cout << "  " << host_gradients[i];
        }

        delete[] host_gradients;

        std::cout << "\n";

        float* h_dC_da = new float[dC_da.size()];
        m_context.transfer(dC_da, h_dC_da);

        for (size_t i = 0; i < dC_da.size(); i++) {
            std::cout << "  " << h_dC_da[i];
        }

        delete[] h_dC_da;

        std::cout << "\n";

        Matrix hidden_gradients(last_hidden_layer.weights.rows(), last_hidden_layer.weights.columns());
        Matrix next_dC_da(m_batchSize, m_layers.at(m_layers.size() - 3).activations.columns());

        m_context.computeGradients(dC_da, last_hidden_layer.activations, last_hidden_layer.activation, hidden_gradients, next_dC_da);

        float* host_gradients_2 = new float[hidden_gradients.size()];
        m_context.transfer(hidden_gradients, host_gradients_2);

        for (size_t i = 0; i < hidden_gradients.size(); i++) {
            std::cout << "  " << host_gradients_2[i];
        }

        delete[] host_gradients_2;

        std::cout << "\n";

        float* h_dC_da_2 = new float[next_dC_da.size()];
        m_context.transfer(next_dC_da, h_dC_da_2);

        for (size_t i = 0; i < next_dC_da.size(); i++) {
            std::cout << "  " << h_dC_da_2[i];
        }

        delete[] h_dC_da_2;

        std::cout << "\n";

        // for (size_t layer = m_layers.size() - 2; layer > 0; layer--) {
        //     Layer& hidden_layer = m_layers[layer];

        //     Matrix gradients();
        // }

        // print the layer logits and activations

        std::cout << "input layer\n";

        float* inputs = new float[m_layers[0].activations.size()];
        m_context.transfer(m_layers[0].activations, inputs);

        for (size_t i = 0; i < m_layers[0].activations.size(); i++) {
            std::cout << "  " << inputs[i];
        }
        std::cout << "\n";

        delete[] inputs;

        std::cout << "hidden layer 1\n";

        float* h1_logits = new float[m_layers[1].logits.size()];
        m_context.transfer(m_layers[1].logits, h1_logits);

        for (size_t i = 0; i < m_layers[1].logits.size(); i++) {
            std::cout << "  " << h1_logits[i];
        }
        std::cout << "\n";

        delete[] h1_logits;

        float* h1_activations = new float[m_layers[1].activations.size()];
        m_context.transfer(m_layers[1].activations, h1_activations);

        for (size_t i = 0; i < m_layers[1].activations.size(); i++) {
            std::cout << "  " << h1_activations[i];
        }
        std::cout << "\n";

        delete[] h1_activations;

        // float* h2_logits = new float[m_layers[2].logits.size()];
        // m_context.transfer(m_layers[2].logits, h2_logits);

        // for (size_t i = 0; i < m_layers[2].logits.size(); i++) {
        //     std::cout << "  " << h2_logits[i];
        // }
        // std::cout << "\n";

        // delete[] h2_logits;

        // float* h1_activations = new float[m_layers[2].activations.size()];
        // m_context.transfer(m_layers[2].activations, h2_logits);

        // for (size_t i = 0; i < m_layers[2].activations.size(); i++) {
        //     std::cout << "  " << h2_logits[i];
        // }
        // std::cout << "\n";

        // delete[] h2_logits;

        std::cout << "output layer\n";

        float* out_logits = new float[m_layers[2].logits.size()];
        m_context.transfer(m_layers[2].logits, out_logits);

        for (size_t i = 0; i < m_layers[2].logits.size(); i++) {
            std::cout << "  " << out_logits[i];
        }
        std::cout << "\n";

        delete[] out_logits;

        float* out_activations = new float[m_layers[2].activations.size()];
        m_context.transfer(m_layers[2].activations, out_activations);

        for (size_t i = 0; i < m_layers[2].activations.size(); i++) {
            std::cout << "  " << out_activations[i];
        }
        std::cout << "\n";

        delete[] out_activations;
    }
}