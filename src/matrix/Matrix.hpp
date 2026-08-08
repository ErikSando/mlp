#pragma once

#include <cstddef>

namespace mlp {
    struct DeviceInt {
        DeviceInt(const int value = 0);
        ~DeviceInt();

        int* data() { return m_data; }

        private:

        int* m_data = nullptr;
    };

    struct DeviceFloat {
        DeviceFloat(const float value = 0.0f);
        ~DeviceFloat();

        float* data() { return m_data; }

        private:

        float* m_data = nullptr;
    };
}