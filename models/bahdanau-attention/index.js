/**
 * Bahdanau's Attention Paper: https://arxiv.org/abs/1409.0473
 *
 * More info: https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05
 */
const tf = require("@tensorflow/tfjs");
const dateFormat = require("../../dataset/date_format");

function createModel(
  inputVocabSize,
  outputVocabSize,
  inputLength,
  outputLength,
  alignmentType = "dot"
) {
  const embeddingDims = 64;
  const lstmUnits = 64;

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
  let encoderGRUOutput = tf.layers
    .bidirectional({
      layer: tf.layers.gru({
        units: lstmUnits,
        returnSequences: true,
        name: "encoderLSTM"
      }) // `returnSequences` returns all the states. Not only the last one
    })
    .apply(encoderEmbeddingOutput);
}

module.exports = { createModel };
