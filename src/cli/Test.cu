

#include <cuda_runtime.h>

#include "cli/CLI.hpp"
#include "cli/Test.hpp"

namespace mlp {
    // void Test(MLP& model, Dataset& dataset) {
    //     // test_data.reset();
    //     dataset.resetLine();

        // size_t input_count = model.getInputCount();
        // size_t batch_size = model.getBatchSize();
        // size_t total_batches = dataset.size() / batch_size;

    //     Matrix correct(1, 1);
    //     Matrix incorrect(1, 1);
    //     Matrix classifications(batch_size, 1);

    //     for (size_t b = 0; b < total_batches; b++) {
    //         Batch batch(batch_size, input_count);
    //         dataset.parseBatch(batch);
    //         model.forwardPass(batch);
    //         model.checkOutputs(batch.labels, batch_size, correct, classifications);

    //         // float* host_outputs = new float[batch_size * 10];
    //         // float* host_classifications = new float[batch_size];

    //         // model.copyOutputs(host_outputs);
    //         // cudaMemcpy(host_classifications, classifications.data(), classifications.size() * sizeof(float), cudaMemcpyDeviceToHost);

    //         // for (int i = 0; i < batch_size; i++) {
    //         //     for (int j = 0; j < 10; j++) {
    //         //         std::cout << "  " << host_outputs[i * 10 + j];
    //         //     }

    //         //     std::cout << "\n";
    //         // }

    //         // std::cout << "\n";

    //         // for (int i = 0; i < batch_size; i++) {
    //         //     std::cout << "  " << host_classifications[i];
    //         // }

    //         // std::cout << "\n";

    //         // for (int i = 0; i < batch_size; i++) {
    //         //     std::cout << "  " << batch.labels[i];
    //         // }

    //         // std::cout << "\n";

    //         // delete[] host_outputs;
    //         // delete[] host_classifications;
    //     }

    //     float host_correct[0];

    //     cudaMemcpy(host_correct, correct.data(), sizeof(float), cudaMemcpyDeviceToHost);

    //     std::cout << "Correct: " << host_correct[0] << "\n";
    // }
}