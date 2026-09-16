#include "context/Context.hpp"
#include "opencl/ContextImpl.hpp"
#include "opencl/impl/Enums.h"

namespace mlp {
    void Context::computeGradients(
        const Matrix& dC_da,
        const Matrix& left_activations, const Matrix& right_activations,
        const Matrix& weights,
        const Activation activation,
        Matrix& weight_gradients, Matrix& bias_gradients, Matrix& dC_da_next
    ) const {
        assert(left_activations.rows() == right_activations.rows());
        assert(weights.size() == left_activations.columns() * right_activations.columns());
        assert(dC_da.size() == right_activations.size());
        assert(dC_da_next.size() == left_activations.size());
        assert(weight_gradients.size() == weights.size());
        assert(bias_gradients.size() == right_activations.columns());

        CLActivation cl_activation = NONE;

        switch (activation) {
            case Activation::LEAKY_RELU:    cl_activation = LEAKY_RELU;break;
            case Activation::RELU:          cl_activation = RELU; break;
            case Activation::SIGMOID:       cl_activation = SIGMOID; break;
            case Activation::TANH:          cl_activation = TANH; break;
        }

        Kernel_up& kernel = m_impl->kernels[opencl::KernelID::BACKPROP_HIDDEN];

        m_impl->setKernelArg(kernel, 0, dC_da.data());
        m_impl->setKernelArg(kernel, 1, left_activations.data());
        m_impl->setKernelArg(kernel, 2, right_activations.data());
        m_impl->setKernelArg(kernel, 3, weights.data());
        m_impl->setKernelArg(kernel, 4, static_cast<unsigned int>(left_activations.columns()));
        m_impl->setKernelArg(kernel, 5, static_cast<unsigned int>(right_activations.columns()));
        m_impl->setKernelArg(kernel, 6, static_cast<unsigned int>(left_activations.rows()));
        m_impl->setKernelArg(kernel, 7, weight_gradients.data());
        m_impl->setKernelArg(kernel, 8, bias_gradients.data());
        m_impl->setKernelArg(kernel, 9, dC_da_next.data());
        m_impl->setKernelArg(kernel, 10, cl_activation);

        size_t lws[3] = { 8, 8, 8 };

        size_t gws[3] = {
            global_work_size(lws[0], left_activations.columns()),
            global_work_size(lws[1], right_activations.columns()),
            global_work_size(lws[2], left_activations.rows())
        };

        if (m_profiler) m_profiler->startTask("Compute Hidden Layer Gradients");

        cl_int err = clEnqueueNDRangeKernel(m_impl->commandQueue, kernel->getCLKernel(), 3, nullptr, gws, lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) CL_ERROR(err, "Failed to enqueue hidden layer back propagation kernel");

        if (m_profiler) m_profiler->endTask("Compute Hidden Layer Gradients");
    }

    void Context::computeOutputGradients(
        const Matrix& last_hidden_activations, const Matrix& output_activations,
        const Matrix& weights,
        const std::vector<int>& labels,
        const OALP al_pair,
        Matrix& weight_gradients, Matrix& bias_gradients, Matrix& dC_da_hidden
    ) const {
        assert(last_hidden_activations.rows() == output_activations.rows());
        assert(weights.size() == last_hidden_activations.columns() * output_activations.columns());
        assert(dC_da_hidden.size() == last_hidden_activations.size());
        assert(weight_gradients.size() == weights.size());
        assert(bias_gradients.size() == output_activations.columns());

        CLActivation cl_activation = NONE;
        CLLoss cl_loss = CCE;

        switch (al_pair) {
            case OALP::NONE_MSE:       cl_loss = MSE; break;
            case OALP::SOFTMAX_CCE:    cl_activation = SOFTMAX; break;
            case OALP::SOFTMAX_MSE:    cl_activation = SOFTMAX; cl_loss = MSE; break;
            default: break;
        }

        Buffer device_labels = createBuffer(labels.size() * sizeof(int));
        transfer(device_labels, (void*) labels.data());

        Kernel_up& kernel = m_impl->kernels[opencl::KernelID::BACKPROP_OUTPUT];

        m_impl->setKernelArg(kernel, 0, last_hidden_activations.data());
        m_impl->setKernelArg(kernel, 1, output_activations.data());
        m_impl->setKernelArg(kernel, 2, weights.data());
        m_impl->setKernelArg(kernel, 3, static_cast<unsigned int>(last_hidden_activations.columns()));
        m_impl->setKernelArg(kernel, 4, static_cast<unsigned int>(output_activations.columns()));
        m_impl->setKernelArg(kernel, 5, static_cast<unsigned int>(last_hidden_activations.rows()));
        m_impl->setKernelArg(kernel, 6, device_labels.data());
        m_impl->setKernelArg(kernel, 7, weight_gradients.data());
        m_impl->setKernelArg(kernel, 8, bias_gradients.data());
        m_impl->setKernelArg(kernel, 9, dC_da_hidden.data());
        m_impl->setKernelArg(kernel, 10, cl_activation);
        m_impl->setKernelArg(kernel, 11, cl_loss);

        size_t lws[3] = { 8, 8, 8 };

        size_t gws[3] = {
            global_work_size(lws[0], last_hidden_activations.columns()),
            global_work_size(lws[1], output_activations.columns()),
            global_work_size(lws[2], last_hidden_activations.rows())
        };

        if (m_profiler) m_profiler->startTask("Compute Output Layer Gradients");

        cl_int err = clEnqueueNDRangeKernel(m_impl->commandQueue, kernel->getCLKernel(), 3, nullptr, gws, lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) CL_ERROR(err, "Failed to enqueue output layer back propagation kernel");

        if (m_profiler) m_profiler->endTask("Compute Output Layer Gradients");
    }

    void Context::optimiseLayer(Matrix& weights, Matrix& biases, const Matrix& weight_gradients, const Matrix& bias_gradients, const float learning_rate) const {
        assert(weights.size() == weight_gradients.size());
        assert(biases.size() == bias_gradients.size());

        Kernel_up& kernel = m_impl->kernels[opencl::KernelID::OPTIMISE_LAYER];

        m_impl->setKernelArg(kernel, 0, weights.data());
        m_impl->setKernelArg(kernel, 1, biases.data());
        m_impl->setKernelArg(kernel, 2, weight_gradients.data());
        m_impl->setKernelArg(kernel, 3, bias_gradients.data());
        m_impl->setKernelArg(kernel, 4, static_cast<unsigned int>(weights.size()));
        m_impl->setKernelArg(kernel, 5, static_cast<unsigned int>(biases.size()));
        m_impl->setKernelArg(kernel, 6, learning_rate);

        size_t lws = LOCAL_WORK_SIZE;
        size_t gws = global_work_size(lws, weights.size());

        if (m_profiler) m_profiler->startTask("Optimise Layer");

        cl_int err = clEnqueueNDRangeKernel(m_impl->commandQueue, kernel->getCLKernel(), 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) CL_ERROR(err, "Failed to enqueue optimise layer kernel");

        if (m_profiler) m_profiler->endTask("Optimise Layer");
    }
}