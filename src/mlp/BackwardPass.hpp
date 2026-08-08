#pragma once

namespace mlp {
    template<typename TDeviceContext>
    void MLP<TDeviceContext>::backwardPass(const Batch& batch) {
        using Layer = Layer<TDeviceContext>;
        using Matrix = typename TDeviceContext::Matrix;

        Layer& output_layer = m_layers.back();
        Layer& last_hidden_layer = m_layers.at(m_layers.size() - 2);

        std::vector<Matrix> gradients; // ordered backwards
        std::vector<Matrix> dC_da;

        gradients.emplace_back(output_layer.weights.rows(), output_layer.weights.columns()).zero();
        dC_da.emplace_back(m_batchSize, last_hidden_layer.activations.columns()).zero();

        // Matrix gradients(output_layer.weights.rows(), output_layer.weights.columns());
        // Matrix dC_da(m_batchSize, last_hidden_layer.activations.columns());

        m_context.computeOutputGradients(
            last_hidden_layer.activations, output_layer.activations, output_layer.weights,
            last_hidden_layer.activations.columns(),
            batch.labels,
            output_layer.activation, Loss::CCE,
            gradients.front(), dC_da.front()
        ); // loss set to CCE for now

        for (size_t l = m_layers.size() - 2; l > 0; l--) {
            Layer& layer = m_layers[l];
            Layer& preceding_layer = m_layers[l - 1];

            gradients.emplace_back(layer.weights.rows(), layer.weights.columns()).zero();
            dC_da.emplace_back(m_batchSize, preceding_layer.activations.columns()).zero();

            m_context.computeGradients(dC_da[dC_da.size() - 2], preceding_layer.activations, layer.activations, layer.weights, layer.activation, gradients.back(), dC_da.back());
        }

        // for (int i = gradients.size() - 1; i >= 0; i--) {
        //     std::cout << "Gradients:\n";

        //     Matrix& grads = gradients[i];

        //     float* host_gradients = new float[grads.size()];

        //     m_context.transfer(grads, host_gradients);

        //     for (size_t j = 0; j < grads.size(); j++) {
        //         std::cout << "  " << host_gradients[j];
        //     }

        //     delete[] host_gradients;

        //     std::cout << "\n\n";
        // }

        for (int i = gradients.size() - 1; i >= 0; i--) {
            Matrix& grads = gradients[i];
            Layer& layer = m_layers[i + 1];

            m_context.optimiseLayer(layer.weights, grads, 0.02f);
            // m_context.optimiseLayer(layer.weights, grads, m_learningRate);
        }

        std::cout << "input layer\n";

        float* inputs = new float[m_layers[0].activations.size()];
        m_context.transfer(m_layers[0].activations, inputs);

        for (size_t i = 0; i < m_layers[0].activations.size(); i++) {
            std::cout << "  " << inputs[i];
        }
        std::cout << "\n";

        delete[] inputs;

        std::cout << "hidden layer 1\n";

        float* h1_activations = new float[m_layers[1].activations.size()];
        m_context.transfer(m_layers[1].activations, h1_activations);

        for (size_t i = 0; i < m_layers[1].activations.size(); i++) {
            std::cout << "  " << h1_activations[i];
        }
        std::cout << "\n";

        delete[] h1_activations;

        std::cout << "output layer\n";

        float* out_activations = new float[m_layers[2].activations.size()];
        m_context.transfer(m_layers[2].activations, out_activations);

        for (size_t i = 0; i < m_layers[2].activations.size(); i++) {
            std::cout << "  " << out_activations[i];
        }
        std::cout << "\n";

        delete[] out_activations;
    }
}