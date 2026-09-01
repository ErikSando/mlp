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

// #pragma once

// namespace mlp {
//     template<typename TContext>
//     void MLP<TContext>::backwardPass(const Batch& batch) {
//         assert(m_batchSize == batch.size);
//         assert(m_batchSize == batch.labels.size());

//         using Layer_t_up = std::unique_ptr<Layer_t>;

//         Layer_t_up& output_layer = m_layers.back();
//         Layer_t_up& last_hidden_layer = m_layers.at(m_layers.size() - 2);

//         std::vector<Matrix_t> gradients; // ordered backwards
//         std::vector<Matrix_t> dC_da;

//         gradients.emplace_back(output_layer->weights.rows(), output_layer->weights.columns()).zero();
//         dC_da.emplace_back(m_batchSize, last_hidden_layer->activations.columns()).zero();

//         m_context.computeOutputGradients(
//             last_hidden_layer->activations, output_layer->activations,
//             output_layer->weights,
//             batch.labels,
//             output_layer->activation, Loss::CCE,
//             gradients.front(), dC_da.front()
//         ); // loss set to CCE for now

//         std::cout << "Weights:";

//         for (Layer_t_up& layer : m_layers) {
//             Matrix_t& weights = layer->weights;
//             float* host_weights = new float[weights.size()];

//             m_context.transfer(host_weights, weights);

//             for (size_t i = 0; i < weights.size(); i++) {
//                 std::cout << "  " << host_weights[i];
//             }
//             std::cout << "\n";

//             delete[] host_weights;
//         }
//         std::cout << "\n";

//         float* host_gradients = new float[gradients.back().size()];
//         m_context.transfer(host_gradients, gradients.back());

//         // std::cout << "Output gradients:\n";
//         // for (size_t i = 0; i < gradients.back().size(); i++) {
//         //     std::cout << "  " << host_gradients[i];
//         // }
//         // std::cout << "\n\n";

//         float* host_dC_da = new float[dC_da.back().size()];
//         m_context.transfer(host_dC_da, dC_da.back());

//         // std::cout << "Hidden layer dC/da:\n";
//         // for (size_t i = 0; i < dC_da.back().size(); i++) {
//         //     std::cout << "  " << host_dC_da[i];
//         // }
//         // std::cout << "\n\n";

//         delete[] host_gradients;
//         delete[] host_dC_da;
    
//         for (size_t l = m_layers.size() - 2; l > 0; l--) {
//             Layer_t_up& layer = m_layers[l];
//             Layer_t_up& preceding_layer = m_layers[l - 1];

//             gradients.emplace_back(layer->weights.rows(), layer->weights.columns()).zero();
//             dC_da.emplace_back(m_batchSize, preceding_layer->activations.columns()).zero();

//             m_context.computeGradients(
//                 dC_da[dC_da.size() - 2],
//                 preceding_layer->activations, layer->activations,
//                 layer->weights,
//                 layer->activation,
//                 gradients.back(), dC_da.back()
//             );

//             float* host_gradients = new float[gradients.back().size()];
//             m_context.transfer(host_gradients, gradients.back());

//             // std::cout << "Gradients:\n";
//             // for (size_t i = 0; i < gradients.back().size(); i++) {
//             //     std::cout << "  " << host_gradients[i];
//             // }
//             // std::cout << "\n\n";

//             delete[] host_gradients;
//         }

//         for (int i = gradients.size() - 1; i >= 0; i--) {
//             Matrix_t& grads = gradients[i];
//             Layer_t_up& layer = m_layers[m_layers.size() - 1 - i];

//             m_context.optimiseLayer(layer->weights, grads, m_learningRate);
//         }

//         // std::cout << "input layer\n";

//         float* inputs = new float[m_layers[0]->activations.size()];
//         m_context.transfer(inputs, m_layers[0]->activations);

//         for (size_t i = 0; i < m_layers[0]->activations.size(); i++) {
//             std::cout << "  " << inputs[i];
//         }
//         std::cout << "\n";

//         delete[] inputs;

//         // std::cout << "hidden layer 1\n";

//         float* h1_activations = new float[m_layers[1]->activations.size()];
//         m_context.transfer(h1_activations, m_layers[1]->activations);

//         for (size_t i = 0; i < m_layers[1]->activations.size(); i++) {
//             std::cout << "  " << h1_activations[i];
//         }
//         std::cout << "\n";

//         delete[] h1_activations;

//         // std::cout << "output layer\n";

//         float* out_activations = new float[m_layers[2]->activations.size()];
//         m_context.transfer(out_activations, m_layers[2]->activations);

//         for (size_t i = 0; i < m_layers[2]->activations.size(); i++) {
//             std::cout << "  " << out_activations[i];
//         }
//         std::cout << "\n\n";

//         delete[] out_activations;

//         // std::cout << "Weights:\n";
//         // for (size_t l = 1; l < m_layers.size(); l++) {
//         //     Layer_t_up& layer = m_layers[l];
//         //     float* weights = new float[layer->weights.size()];
//         //     m_context.transfer(weights, layer->weights);

//         //     for (size_t i = 0; i < layer->weights.size(); i++) {
//         //         std::cout << "  " << weights[i];
//         //     }
//         //     std::cout << "\n";

//         //     delete[] weights;
//         // }
//         // std::cout << "\n";
//     }
// }