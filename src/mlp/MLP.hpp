#pragma once

#include "device/DeviceContext.hpp"
#include "matrix/Matrix.hpp"
#include "mlp/Batch.hpp"

// things are very experimental right now

namespace mlp {
    class MLP {
        public:

        MLP(DeviceContext& device_context, std::vector<size_t>& layer_sizes, const size_t batch_size);
        ~MLP();

        void forwardPass(Batch& batch);

        private:

        DeviceContext& m_context;

        size_t m_batchSize;
        size_t m_inputCount;

        std::vector<Matrix> m_weights;
        std::vector<Matrix> m_layers; // node values, includes hidden layers and the output layer
    };
}