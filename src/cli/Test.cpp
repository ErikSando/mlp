#include "cli/CLI.hpp"
#include "cli/Commands.hpp"

namespace mlp {
    namespace commands {
        void test(MLP_t* model, Dataset& dataset, TestData& data) {
            data.reset();
            dataset.resetLine();

            const Context& context = model->getContext();

            size_t input_count = model->getInputCount();
            size_t batch_size = model->getBatchSize();
            size_t n_batches = dataset.size() / batch_size;
            size_t remainder = dataset.size() - n_batches * batch_size;

            Buffer correct(sizeof(int));
            Buffer classifications(batch_size * sizeof(int));

            correct.zero();
            classifications.zero();

            Batch batch(batch_size, input_count);

            for (size_t b = 0; b < n_batches; b++) {
                dataset.parseBatch(batch);
                model->forwardPass(batch);
                model->checkOutputs(batch.labels, correct, classifications);
            }

            dataset.parseBatch(batch);
            model->forwardPass(batch);
            model->checkOutputs(batch.labels, correct, classifications, remainder);

            int host_correct;

            context.transfer((void*) &host_correct, correct);

            size_t total = n_batches * batch_size + remainder;
            size_t correct_ul = (size_t) host_correct;
            size_t incorrect = total - correct_ul;

            data.correct = correct_ul;
            data.incorrect = incorrect;
        }
    }
}