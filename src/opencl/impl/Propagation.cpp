#include "context/Context.hpp"
#include "opencl/ContextImpl.hpp"
#include "opencl/impl/Config.h"
#include "opencl/impl/Enums.h"
#include "opencl/Kernel.hpp"

namespace mlp {
    void Context::propagate(
        const Matrix& left_activations,
        Matrix& right_logits, Matrix& right_activations,
        const Matrix& weights, const Matrix& biases,
        const Activation activation
    ) const {
        assert(left_activations.columns() == weights.rows());
        assert(right_activations.rows() == left_activations.rows());
        assert(right_activations.columns() == weights.columns());
        assert(right_activations.size() == right_logits.size());

        cl_int err;

        Kernel_up& propagate_kernel = m_impl->kernels[opencl::KernelID::PROPAGATION];

        CLActivation cl_activation = NONE;

        switch (activation) {
            case Activation::LEAKY_RELU:    cl_activation = LEAKY_RELU;break;
            case Activation::RELU:          cl_activation = RELU; break;
            case Activation::SIGMOID:       cl_activation = SIGMOID; break;
            case Activation::TANH:          cl_activation = TANH; break;
        }

        // err = propagate_kernel->setArgument(0, left_activations.data());
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 0");

        // err = propagate_kernel->setArgument(1, right_logits.data());
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 1");
    
        // err = propagate_kernel->setArgument(2, right_activations.data());
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 2");

        // err = propagate_kernel->setArgument(3, weights.data());
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 3");

        // err = propagate_kernel->setArgument(4, biases.data());
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 4");

        // err = propagate_kernel->setArgument(5, static_cast<unsigned int>(right_logits.rows()));
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 5");

        // err = propagate_kernel->setArgument(6, static_cast<unsigned int>(left_activations.columns()));
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 6");

        // err = propagate_kernel->setArgument(7, static_cast<unsigned int>(right_logits.columns()));
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 7");

        // err = propagate_kernel->setArgument(8, cl_activation);
        // if (err != CL_SUCCESS) CL_ERROR(err, "Failed to set kernel argument 8");

        m_impl->setKernelArg(propagate_kernel, 0, left_activations.data());
        m_impl->setKernelArg(propagate_kernel, 1, right_logits.data());
        m_impl->setKernelArg(propagate_kernel, 2, right_activations.data());
        m_impl->setKernelArg(propagate_kernel, 3, weights.data());
        m_impl->setKernelArg(propagate_kernel, 4, biases.data());
        m_impl->setKernelArg(propagate_kernel, 5, static_cast<unsigned int>(right_logits.rows()));
        m_impl->setKernelArg(propagate_kernel, 6, static_cast<unsigned int>(left_activations.columns()));
        m_impl->setKernelArg(propagate_kernel, 7, static_cast<unsigned int>(right_logits.columns()));
        m_impl->setKernelArg(propagate_kernel, 8, cl_activation);

        size_t lws[2] = { LOCAL_WORK_SIZE, LOCAL_WORK_SIZE };

        size_t gws[2] = {
            global_work_size(lws[0], right_activations.columns()),
            global_work_size(lws[1], right_activations.rows())
        };

        if (m_profiler) m_profiler->startTask("Propagation");

        err = clEnqueueNDRangeKernel(m_impl->commandQueue, propagate_kernel->getCLKernel(), 2, nullptr, gws, lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) CL_ERROR(err, "Failed to enqueue propagate kernel");

        if (activation == Activation::SOFTMAX) {
            size_t lws = SOFTMAX_WORK_GROUP_SIZE;
            size_t gws = lws * right_logits.rows();

            Kernel_up& softmax_kernel = m_impl->kernels[opencl::KernelID::SOFTMAX];

            m_impl->setKernelArg(softmax_kernel, 0, right_logits.data());
            m_impl->setKernelArg(softmax_kernel, 1, right_activations.data());
            m_impl->setKernelArg(softmax_kernel, 2, static_cast<unsigned int>(right_logits.rows()));
            m_impl->setKernelArg(softmax_kernel, 3, static_cast<unsigned int>(right_logits.columns()));

            if (m_profiler) m_profiler->startTask("Softmax");

            err = clEnqueueNDRangeKernel(m_impl->commandQueue, softmax_kernel->getCLKernel(), 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) CL_ERROR(err, "Failed to enqueue softmax kernel");

            if (m_profiler) m_profiler->endTask("Softmax");
        }

        if (m_profiler) m_profiler->endTask("Propagation");
    }
}