// #pragma once

// #include <cstddef>

// namespace mlp {
//     struct TestData {
//         void reset() {
//             incorrect = 0UL;
//             correct = 0UL;
//         }

//         float getAccuracy() { return (float) correct / (float) (correct + incorrect); }

//         size_t incorrect;
//         size_t correct;
//     };

//     namespace commands {
//         template<typename TContext>
//         void test(MLP<TContext>& model, Context& context, Dataset& dataset, TestData& data) {
//             data.reset();
//             dataset.resetLine();

//             size_t input_count = model.getInputCount();
//             size_t batch_size = model.getBatchSize();
//             size_t total_batches = dataset.size() / batch_size;

//             Matrix correct(1, 1);
//             Matrix classifications(batch_size, 1);

//             Batch batch(batch_size, input_count);

//             for (size_t b = 0; b < total_batches; b++) {
//                 dataset.parseBatch(batch);
//                 model.forwardPass(batch);
//                 model.checkOutputs(batch.labels, batch_size, correct, classifications);
//             }

//             float host_correct;

//             context.transfer(host_correct, correct);

//             size_t total = total_batches * batch_size;
//             size_t correct_ul = (size_t) host_correct;
//             size_t incorrect = total - correct_ul;

//             data.correct = correct_ul;
//             data.incorrect = incorrect;
//         }
//     }
// }