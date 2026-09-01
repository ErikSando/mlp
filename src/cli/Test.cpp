#include "cli/CLI.hpp"
#include "cli/Commands.hpp"

namespace mlp {
    namespace commands {
        void test(MLP_t* model, Context& context, Dataset& dataset, TestData& data) {
            data.reset();
            dataset.resetLine();

            size_t input_count = model->getInputCount();
            size_t batch_size = model->getBatchSize();
            size_t total_batches = dataset.size() / batch_size;
            // size_t total_batches = 10;

            Buffer correct(sizeof(int));
            Buffer classifications(batch_size * sizeof(int));

            correct.zero();
            classifications.zero();

            Batch batch(batch_size, input_count);

            for (size_t b = 0; b < total_batches; b++) {
                dataset.parseBatch(batch);
                model->forwardPass(batch);
                model->checkOutputs(batch.labels, correct, classifications);

                // int* host_classifications = new int[batch_size];

                // context.transfer((void*) host_classifications, classifications);

                // std::cout << host_classifications[0];
                // for (size_t i = 1; i < batch_size; i++) {
                //     std::cout << ", " <<  host_classifications[i];
                // }
                // std::cout << "\n";

                // delete[] host_classifications;

                // if (b % 10 == 9) {
                //     int host_correct;

                //     context.transfer((void*) &host_correct, correct);

                //     int total = b * batch_size;

                //     std::cout << "Accuracy: " << (float) host_correct /  (float) total * 100 << "\n";
                // }
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