#pragma once

#include "device/DeviceContext.hpp"
#include "matrix/Matrix.hpp"
#include "mlp/InputLayer.hpp"

// things are very experimental right now

namespace mlp {
    class MLP {
        public:

        MLP(DeviceContext& device_context, std::vector<size_t>& layer_sizes);
        ~MLP();

        void forwardPass(InputLayer& input_layer);

        private:

        DeviceContext& m_context;

        std::vector<Matrix> m_weights;
        std::vector<Matrix> m_layers; // node values, includes hidden layers and the output layer
    };
}