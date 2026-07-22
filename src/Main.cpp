#include <iostream>

#include "cuda/Interface.hpp"
#include "matrix/Matrix.hpp"

int countDigits(int n) {
    int chars = 1;

    while (std::abs(n) >= 10) {
        n /= 10;
        chars++;
    }

    return chars;
}

void printMatrix(float* matrix, int rows, int columns, int padding = 5) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {
            int index = c + r * columns;

            int value = matrix[index];
            if (value >= 0) std::cout << " ";

            std::cout << value;

            int digits = countDigits(value);
            for (int i = 0; i < padding - digits; i++) std::cout << " ";
        }

        std::cout << "\n";
    }
}

int main() {
    mlp::Matrix mA(5, 4);
    mlp::Matrix mB(4, 5);
    mlp::Matrix mC(5, 5);

    mlp::Interface interface;

    interface.randomise(mA);
    interface.randomise(mB);

    float dataA[mA.size()];
    float dataB[mB.size()];

    interface.transfer(mA, dataA);
    interface.transfer(mB, dataB);

    printMatrix(dataA, mA.rows(), mA.columns(), 4);
    std::cout << "\n";
    printMatrix(dataB, mB.rows(), mB.columns(), 4);
    std::cout << "\n";

    interface.multiply(mA, mB, mC);

    float dataC[mC.size()];

    interface.transfer(mC, dataC);

    printMatrix(dataC, mC.rows(), mC.columns());

    return 0;
}