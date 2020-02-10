/**
 * Sequence-2-Sequence
 */

const tf = require("@tensorflow/tfjs");
const dateFormat = require("../../dataset/date_format");

function createModel(
  inputVocabSize,
  outputVocabSize,
  inputLength, // Encoder unrollings
  outputLength // Decoder unrollings
) {
  const embeddingDims = 64;
  const lstmUnits = 64;

  /** ENCODER */
  const encoderEmbeddingInput = tf.input({
    shape: [inputLength],
    name: "embeddingEncoderInput"
  });

  // Select the embedding vectors by providing the `encoderEmbeddingInput` with the indices
  let encoderEmbeddingOutput = tf.layers
    .embedding({
      inputDim: inputVocabSize,
      outputDim: embeddingDims,
      inputLength: inputLength,
      maskZero: true,
      name: "encoderEmbedding"
    })
    .apply(encoderEmbeddingInput);

  // Feed the selected embedding vectors to the LSTM
  let encoderLSTMOutput = tf.layers
    .lstm({ units: lstmUnits, returnSequences: false, name: "encoderLSTM" }) // `returnSequences` returns all the states. Not only the last one
    .apply(encoderEmbeddingOutput);

  /** DECODER */
  const decoderEmbeddingInput = tf.input({
    shape: [outputLength],
    name: "embeddingDecoderInput"
  });
  // Select the embedding vectors by providing the `decoderEmbeddingInput` with the indices
  let decoderEmbeddingOutput = tf.layers
    .embedding({
      inputDim: outputVocabSize,
      outputDim: embeddingDims,
      inputLength: outputLength,
      maskZero: true,
      name: "decoderEmbedding"
    })
    .apply(decoderEmbeddingInput);

  // Feed the selected embedding vectors to the LSTM
  let decoderLSTMOutput = tf.layers
    .lstm({ units: lstmUnits, returnSequences: true, name: "decoderLSMT" })
    .apply(decoderEmbeddingOutput, {
      initialState: [encoderLSTMOutput, encoderLSTMOutput]
    });

  // Input is [null, 10, 64]
  // Outputs is [null, 10, 13] outputVocabSize = 13
  outputGenerator = tf.layers
    .timeDistributed({
      layer: tf.layers.dense({
        units: outputVocabSize,
        activation: "softmax",
        name: "timeDistributedSoftmax"
      }) // Generate the probability of the output char
    })
    .apply(decoderLSTMOutput);

  const model = tf.model({
    inputs: [encoderEmbeddingInput, decoderEmbeddingInput],
    outputs: outputGenerator
  });

  model.compile({ loss: "categoricalCrossentropy", optimizer: "adam" });
  return model;
}

module.exports = { createModel };
