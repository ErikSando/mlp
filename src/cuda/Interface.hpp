#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    class Interface {
        public:

        void multiply(Matrix& mA, Matrix& mB, Matrix& mC);
        void transfer(Matrix& src, float* dest);

        void randomise(Matrix& m);
    };
}