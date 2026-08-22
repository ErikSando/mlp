#include "cli/CLI.hpp"
#include "cli/Commands.hpp"

namespace mlp {
    namespace commands {
        void test(MLP_t& model, Context& context, Dataset& dataset, TestData& data) {
            data.reset();
            dataset.resetLine();

            size_t input_count = model.getInputCount();
            size_t batch_size = model.getBatchSize();
            size_t total_batches = dataset.getSize() / batch_size;

            Buffer correct(sizeof(int));
            Buffer classifications(batch_size * sizeof(int));

            correct.zero();
            classifications.zero();

            Batch batch(batch_size, input_count);

            for (size_t b = 0; b < total_batches; b++) {
                dataset.parseBatch(batch);
                model.forwardPass(batch);
                model.checkOutputs(batch.labels, correct, classifications);
            }

            int host_correct;

            context.transfer((void*) &host_correct, correct);

            size_t total = total_batches * batch_size;
            size_t correct_ul = (size_t) host_correct;
            size_t incorrect = total - correct_ul;

            data.correct = correct_ul;
            data.incorrect = incorrect;
        }
    }
}