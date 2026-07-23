#pragma once

#include "nn/Input.hpp"

// things are very experimental right now

namespace mlp {
    class MLP {
        public:

        MLP();
        ~MLP();

        void forwardPass(InputNodes inp);
        void forwardPass(std::vector<InputNodes>& inp);

        private:


    };
}