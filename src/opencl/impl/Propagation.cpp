#include "context/Context.hpp"
#include "opencl/ContextImpl.hpp"
#include "opencl/Kernel.hpp"

namespace mlp {
    void Context::propagate(
        const Matrix& previous_activations,
        Matrix& logits, Matrix& activations,
        const Matrix& weights, const Matrix& biases,
        const Activation activation
    ) const {
        cl_int err;

        using Kernel_up = std::unique_ptr<opencl::Kernel>;

        Kernel_up& propagation_kernel = m_impl->kernels[opencl::KernelID::PROPAGATION];

        err = propagation_kernel->setArgument(0, previous_activations.data());
        err = propagation_kernel->setArgument(0, logits.data());
    }
}