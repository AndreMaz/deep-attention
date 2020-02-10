/**
 * Luong's Attention Paper: https://arxiv.org/abs/1508.04025
 */

const tf = require("@tensorflow/tfjs");

const GetLastTimestepLayer = require("./last-time-step-layer");
tf.serialization.registerClass(GetLastTimestepLayer);

/**
 * Create an LSTM-based attention model for date conversion.
 *
 * @param {number} inputVocabSize Input vocabulary size. This includes
 *   the padding symbol. In the context of this model, "vocabulary" means
 *   the set of all unique characters that might appear in the input date
 *   string.
 * @param {number} outputVocabSize Output vocabulary size. This includes
 *   the padding and starting symbols. In the context of this model,
 *   "vocabulary" means the set of all unique characters that might appear in
 *   the output date string.
 * @param {number} inputLength Maximum input length (# of characters). Input
 *   sequences shorter than the length must be padded at the end.
 * @param {number} outputLength Output length (# of characters).
 * @param {string} alignmentType Type of Luong's alignment score. Can be `dot`, `general` or `concat`.
 * @return {tf.LayersModel} A compiled model instance.
 */
function createModel(
  inputVocabSize,
  outputVocabSize,
  inputLength,
  outputLength,
  alignmentType = "dot"
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
    .lstm({ units: lstmUnits, returnSequences: true, name: "encoderLSTM" }) // `returnSequences` returns all the states. Not only the last one
    .apply(encoderEmbeddingOutput);

  // Slice the `encoderLSTMOutput` to get the last State of encoder's LSTM.
  // It will be used to init the decoder's LSTM
  const encoderLastState = new GetLastTimestepLayer({
    name: "encoderLastStateExtractor"
  }).apply(encoderLSTMOutput);

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
      initialState: [encoderLastState, encoderLastState]
    });

  /** ATTENTION */
  // More info: https://arxiv.org/pdf/1508.04025.pdf
  let attention;
  if (alignmentType === "dot") {
    // Compute hidden_target * hidden_source
    attention = tf.layers
      .dot({ axes: [2, 2], name: "attentionDot" })
      .apply([decoderLSTMOutput, encoderLSTMOutput]);
  } else if (alignmentType === "general") {
    // Compute W_a * hidden_source
    let linearActivation = tf.layers
      .dense({ units: lstmUnits, name: "denseOnEncoder" })
      .apply(encoderLSTMOutput);
    // Compute hidden_target * "linearActivation"
    attention = tf.layers
      .dot({ axes: [2, 2], name: "attentionDot" })
      .apply([decoderLSTMOutput, linearActivation]);
  } else if (alignmentType === "concat") {
    throw new Error("Concat Score Function Not Implemented!");
    /*
    // Concat hidden_target and hidden_source
    let concat = tf.layers
      .concatenate({ name: "concatHiddenStates" })
      .apply([decoderLSTMOutput, encoderLSTMOutput]);

    let linearActivation = tf.layers
      .activation({
        activation: "linear",
        name: "linearActivationOnConcat"
      })
      .apply(concat);

    let tanhed = tf.layers
      .activation({
        activation: "tanh",
        name: "linearActivationOnConcat"
      })
      .apply(linearActivation);

    attention = tf.layers
      .activation({
        activation: "linear",
        name: "linearActivationOnThan"
      })
      .apply(tanhed);
      */
  }

  // Apply soft max activation to map values into probabilities
  // This produces the "Attention Weights"
  attention = tf.layers
    .activation({ activation: "softmax", name: "attentionSoftMax" })
    .apply(attention);

  // Apply "Attention Weights" on each output state of encoder LSTM
  // The result "shows" where (at what part of input) to "focus" on
  // The result is the "Context Vector"
  const context = tf.layers
    .dot({ axes: [2, 1], name: "context" })
    .apply([attention, encoderLSTMOutput]);

  // Concatenates the decoder LSTM Output with the "Context Vector"
  // Context shape:           [null, 10, 64]
  // decoderLSTMOutput shape: [null, 10, 64]
  // Result                   [null, 10, 128]
  const decoderCombinedContext = tf.layers
    .concatenate({ name: "combinedContext" })
    .apply([context, decoderLSTMOutput]);

  // Apply the SAME "dense" layer over `time` dimension.
  // The input should be at least 3D, and the dimension of the index `1` will be considered to be the temporal dimension.
  // More info: https://js.tensorflow.org/api/latest/#layers.timeDistributed
  // Input is [null, 10, 128]. The "null" (sample size) is not considered.
  // This means that the input shape is [10, 128]. Index `1` is 128. So the timeDistributed will be applied over 128
  // Outputs is [null, 10, 64] lstmUnits = 64
  // More info about time distributed layer: https://medium.com/smileinnovation/how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00
  // Another one: https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
  // And another one: https://github.com/keras-team/keras/issues/1029#issuecomment-158064822
  let outputGenerator = tf.layers
    .timeDistributed({
      layer: tf.layers.dense({
        units: lstmUnits,
        activation: "tanh", // Scale all the data between -1 and 1
        name: "timeDistributedTanh"
      })
    })
    .apply(decoderCombinedContext);

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
    .apply(outputGenerator);

  const model = tf.model({
    inputs: [encoderEmbeddingInput, decoderEmbeddingInput],
    outputs: outputGenerator
  });

  model.compile({ loss: "categoricalCrossentropy", optimizer: "adam" });
  return model;
}

module.exports = {
  createModel
};
