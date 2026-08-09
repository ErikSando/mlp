#pragma once

#include <cstring>

#include "data/Sample.hpp"
#include "mlp/MLP.hpp"

namespace mlp {
    template<typename TContext>
    void MLP<TContext>::classify(const Sample& sample, ClassifyInfo& info) {
        size_t n_inputs = m_layers[0]->activations.columns();

        assert(sample.data.size() == n_inputs);

        Batch batch(m_batchSize, n_inputs);

        std::memcpy(batch.data.data(), sample.data.data(), n_inputs * sizeof(float));
        std::memset(&batch.data.at(n_inputs), 0, n_inputs * (m_batchSize - 1));

        forwardPass(batch);

        size_t n_classes = m_layers.back()->activations.columns();
        size_t n_outputs = m_layers.back()->activations.size();

        float* outputs = new float[n_outputs];
        copyOutputs(outputs);

        info.outputs.resize(n_classes);
        std::memcpy(info.outputs.data(), outputs, n_classes * sizeof(float));

        int classification = UNDEFINED_CLASS;
        float max_value = 0.0f;

        for (size_t i = 0; i < n_classes; i++) {
            if (outputs[i] > max_value) {
                max_value = outputs[i];
                classification = i;
            }
        }

        delete[] outputs;

        info.classification = classification;

        if (sample.label == NO_LABEL) return;

        assert(sample.label < n_outputs);

        float* host_targets = new float[n_outputs] {};
        host_targets[sample.label] = 1.0f;

        float* host_results = new float[m_batchSize];

        Matrix_t targets(m_batchSize, n_classes);
        Matrix_t results(m_batchSize, 1);

        m_context.transfer(host_targets, targets);
        m_context.computeLoss(m_layers.back()->activations, targets, results, m_lossFunction);
        m_context.transfer(results, host_results);

        info.error = host_results[0];

        delete[] host_targets;
        delete[] host_results;
    }
}