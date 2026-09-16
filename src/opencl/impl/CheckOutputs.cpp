#include "context/Context.hpp"
#include "opencl/ContextImpl.hpp"

namespace mlp {
    void Context::checkOutputs(const Matrix& outputs, const std::vector<int>& labels, Buffer& correct, Buffer& classifications, const size_t samples) const {
        assert(outputs.rows() == labels.size());
        assert(samples <= outputs.rows());

        Buffer device_labels = createBuffer(samples * sizeof(int));
        transfer(device_labels, (void*) labels.data());

        if (m_profiler) m_profiler->startTask("Check Outputs");

        Kernel_up& kernel = m_impl->kernels[opencl::KernelID::CHECK_OUTPUTS];

        m_impl->setKernelArg(kernel, 0, outputs.data());
        m_impl->setKernelArg(kernel, 1, device_labels.data());
        m_impl->setKernelArg(kernel, 2, static_cast<unsigned int>(samples));
        m_impl->setKernelArg(kernel, 3, static_cast<unsigned int>(outputs.columns()));
        m_impl->setKernelArg(kernel, 4, correct.data());
        m_impl->setKernelArg(kernel, 5, classifications.data());

        size_t lws = LOCAL_WORK_SIZE;
        size_t gws = global_work_size(lws, samples);

        cl_int err = clEnqueueNDRangeKernel(m_impl->commandQueue, kernel->getCLKernel(), 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) CL_ERROR(err, "Failed to enqueue check outputs kernel");

        if (m_profiler) m_profiler->endTask("Check Outputs");
    }
}