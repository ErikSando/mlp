#include <cassert>

#include "mlp/MLP.hpp"

namespace mlp {
    void MLP::forwardPass(Batch& batch) {
        assert(m_batchSize * m_inputCount == batch.data.size());

        Matrix input(m_batchSize, m_inputCount);

        m_context.transfer(batch.data.data(), input);
    }
}