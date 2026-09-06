#ifndef ENUMS_H
#define ENUMS_H

typedef enum {
    NONE, LEAKY_RELU, RELU, SIGMOID, TANH, SOFTMAX
} CLActivation;

typedef enum {
    CCE, MSE
} CLLoss;

#endif