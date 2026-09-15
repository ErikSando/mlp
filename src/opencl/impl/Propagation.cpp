#include "context/Context.hpp"
#include "opencl/ContextImpl.hpp"
#include "opencl/impl/Enums.h"
#include "opencl/Kernel.hpp"

namespace mlp {
    void Context::propagate(
        const Matrix& left_activations,
        Matrix& right_logits, Matrix& right_activations,
        const Matrix& weights, const Matrix& biases,
        const Activation activation
    ) const {
        cl_int err;

        using Kernel_up = std::unique_ptr<opencl::Kernel>;

        Kernel_up& propagation_kernel = m_impl->kernels[opencl::KernelID::PROPAGATION];

        CLActivation cl_activation = NONE;

        switch (activation) {
            case Activation::LEAKY_RELU:
                cl_activation = LEAKY_RELU;
            break;

            case Activation::RELU:
                cl_activation = RELU;
            break;

            case Activation::SIGMOID:
                cl_activation = SIGMOID;
            break;

            case Activation::TANH:
                cl_activation = TANH;
            break;
        }

        err = propagation_kernel->setArgument(0, left_activations.data());

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to set kernel arguments 0");
        }
        err = propagation_kernel->setArgument(1, right_logits.data());

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to set kernel arguments 1");
        }
        err = propagation_kernel->setArgument(2, right_activations.data());

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to set kernel arguments 2");
        }
        err = propagation_kernel->setArgument(3, weights.data());

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to set kernel arguments 3");
        }
        err = propagation_kernel->setArgument(4, biases.data());

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to set kernel arguments 4");
        }
        err = propagation_kernel->setArgument(5, static_cast<int>(right_logits.rows()));

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to set kernel arguments 5");
        }
        err = propagation_kernel->setArgument(6, static_cast<int>(left_activations.columns()));

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to set kernel arguments 6");
        }
        err = propagation_kernel->setArgument(7, static_cast<int>(right_logits.columns()));

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to set kernel arguments 7");
        }
        err = propagation_kernel->setArgument(8, cl_activation);

        if (err != CL_SUCCESS) {
            ERROR("Failed to set kernel arguments");
        }

        size_t lws[2] = { LOCAL_WORK_SIZE, LOCAL_WORK_SIZE };

        size_t gws[2] = {
            global_work_size(lws[0], right_activations.columns()),
            global_work_size(lws[1], right_activations.rows())
        };

        err = clEnqueueNDRangeKernel(m_impl->commandQueue, propagation_kernel->getCLKernel(), 2, nullptr, gws, lws, 0, nullptr, nullptr);

        if (err != CL_SUCCESS) {
            CL_ERROR(err, "Failed to enqueue propagate kernel");
        }
    }
}