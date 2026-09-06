#include <iostream>

#include "opencl/Error.hpp"
#include "opencl/ContextImpl.hpp"
#include "opencl/profiling/Task.hpp"

namespace mlp {
    namespace opencl {
        Task::Task(const cl_command_queue& command_queue) : m_commandQueue(command_queue) {}

        Task::~Task() {
            for (EventPair& pair : m_eventPairs) {
                if (pair.start) clReleaseEvent(pair.start);
                if (pair.end) clReleaseEvent(pair.end);
            }
        }

        void Task::start() {
            cl_event start;
            cl_int err = clEnqueueMarkerWithWaitList(m_commandQueue, 0, nullptr, &start);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to create task start marker");
            }

            m_lastStart = start;
        }

        void Task::end() {
            if (!m_lastStart) {
                ERROR("Tried to end a task that hasn't been started");
            }

            cl_event end;
            cl_int err = clEnqueueMarkerWithWaitList(m_commandQueue, 0, nullptr, &end);

            if (err != CL_SUCCESS) {
                CL_ERROR(err, "Failed to create task end marker");
            }

            m_eventPairs.push_back({ m_lastStart, end });

            m_lastStart = nullptr;
        }

        void Task::resolve() {
            for (EventPair& pair : m_eventPairs) {
                m_count++; // not using one += m_eventPairs.size() so that the || m_count == 1 works for setting the minimum

                clWaitForEvents(1, &pair.end);

                cl_ulong start_time;
                cl_ulong end_time;

                clGetEventProfilingInfo(pair.start, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
                clGetEventProfilingInfo(pair.end, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);

                float duration = static_cast<float>(end_time - start_time) / 1000000.0f;

                if (duration < m_min || m_count == 1) m_min = duration;
                if (duration > m_max) m_max = duration;

                m_duration += duration;

                clReleaseEvent(pair.start);
                clReleaseEvent(pair.end);
            }

            m_eventPairs.clear();

            m_average = m_duration / static_cast<float>(m_count);
        }

        float Task::getDuration() {
            if (m_eventPairs.size()) resolve();
            return m_duration;
        }

        float Task::getMin() {
            if (m_eventPairs.size()) resolve();
            return m_min;
        }

        float Task::getMax() {
            if (m_eventPairs.size()) resolve();
            return m_max;
        }

        float Task::getAverage() {
            if (m_eventPairs.size()) resolve();
            return m_average;
        }
    }
}