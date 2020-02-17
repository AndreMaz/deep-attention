const tf = require("@tensorflow/tfjs");
const DecoderBahdanau = require("./decoder_v2");

function createModel(
  inputVocabSize,
  outputVocabSize,
  inputLength,
  outputLength,
  alignmentType = "dot"
) {
  const embeddingDims = 64;
  const lstmUnits = 32;
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
  }).apply([decoderEmbeddingInput, encoderLSTMOutput]);

  const model = tf.model({
    inputs: [encoderEmbeddingInput, decoderEmbeddingInput],
    outputs: decoderLSTMOutput
  });

  return model;
}

module.exports = { createModel };
