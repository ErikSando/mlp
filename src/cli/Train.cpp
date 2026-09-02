#include "cli/CLI.hpp"
#include "cli/Commands.hpp"

namespace mlp {
    namespace commands {
        void train(MLP_t* model, Dataset& dataset, size_t n_epochs) {
            size_t input_count = model->getInputCount();
            size_t batch_size = model->getBatchSize();
            size_t n_batches = dataset.size() * n_epochs / batch_size;

            Batch batch(batch_size, input_count);

            for (size_t b = 0; b < n_batches; b++) {
                dataset.parseBatch(batch);
                model->forwardPass(batch);
                model->backwardPass(batch);
            }

            // I think I should provide some training insights
        }
    }
}