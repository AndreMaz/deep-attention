const tf = require("@tensorflow/tfjs");
const DecoderBahdanau = require("./decoder_v2");
const GetLastTimestepLayer = require("./last-time-step-layer");

function createModel(
  inputVocabSize,
  outputVocabSize,
  inputLength,
  outputLength
) {
  const embeddingDims = 64;
  const lstmUnits = 64;
  const batchSize = 32;

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
    .lstm({ units: lstmUnits, returnSequences: true, name: "encoderLSTM" }) // `returnSequences` returns all the states. Not only the last one
    .apply(encoderEmbeddingOutput);

  const encoderLastState = new GetLastTimestepLayer({
    name: "encoderLastStateExtractor"
  }).apply(encoderLSTMOutput);

  const decoderEmbeddingInput = tf.input({
    shape: [outputLength],
    name: "embeddingDecoderInput"
  });

  let decoderLSTMOutput = new DecoderBahdanau({
    name: "CustomLSTM",
    outputLength: outputLength,
    outputVocabSize: outputVocabSize,
    embeddingDims: lstmUnits,
    lstmUnits: lstmUnits,
    batchSize: batchSize
  }).apply([decoderEmbeddingInput, encoderLastState, encoderLSTMOutput]);

  // Input is [null, 10, 64]
  // Outputs is [null, 10, 13] outputVocabSize = 13
  let outputGenerator = tf.layers
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

  model.summary();

  model.compile({ loss: "categoricalCrossentropy", optimizer: "adam" });
  return model;
}

module.exports = { createModel };
