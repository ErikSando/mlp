#pragma once

#include "device/DeviceContext.hpp"
// #include "matrix/Matrix.hpp"
#include "mlp/Batch.hpp"
#include "mlp/Layer.hpp"

// things are very experimental right now

namespace mlp {
    class MLP {
        public:

        MLP(DeviceContext& device_context, std::vector<size_t>& layer_sizes, const size_t batch_size);
        ~MLP();

        void forwardPass(Batch& batch);

        // im thinking the forward pass function will have no return type and pass the values through the network
        // and the output values can be read using another function
        // i think this makes sense considering the node values are on the GPU anyway

        // Copy the values of the output nodes into the given host memory location
        void copyOutputs(float* host_outputs);
        // ^^^ use a safer method in the future, the size of host_outputs isnt enforced right now

        private:

        DeviceContext& m_context;

        size_t m_batchSize;
        size_t m_inputCount;

        std::vector<Layer> m_layers;
    };
}