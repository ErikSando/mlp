#include "context/Context.hpp"
#include "opencl/ContextImpl.hpp"

namespace mlp {
    void Context::computeLoss(const Matrix& outputs, const Matrix& targets, Matrix& result, const Loss loss) const {
        assert(outputs.size() == targets.size());
        assert(outputs.rows() == targets.rows());
        assert(outputs.columns() == targets.columns());
        assert(result.size() == outputs.rows());

        size_t id = opencl::KernelID::NONE;

        switch (loss) {
            case Loss::CCE: id = opencl::KernelID::CCE; break;
            case Loss::MSE: id = opencl::KernelID::MSE; break;
            default: break;
        }

        if (id == opencl::KernelID::NONE) ERROR("Invalid loss function");

        Kernel_up& loss_kernel = m_impl->kernels[id];

        m_impl->setKernelArg(loss_kernel, 0, outputs.data());
        m_impl->setKernelArg(loss_kernel, 1, targets.data());
        m_impl->setKernelArg(loss_kernel, 2, static_cast<unsigned int>(outputs.columns()));
        m_impl->setKernelArg(loss_kernel, 3, static_cast<unsigned int>(outputs.rows()));
        m_impl->setKernelArg(loss_kernel, 4, result.data());

        size_t lws = outputs.columns();
        size_t gws = global_work_size(lws, outputs.size());

        if (m_profiler) m_profiler->startTask("Compute Loss");

        cl_int err = clEnqueueNDRangeKernel(m_impl->commandQueue, loss_kernel->getCLKernel(), 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) CL_ERROR(err, "Failed to enqueue loss kernel");

        if (m_profiler) m_profiler->endTask("Compute Loss");
    }
}