/**
 * More info: https://arxiv.org/abs/1409.0473
 *
 * More info: https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05
 */

const tf = require("@tensorflow/tfjs");
const dateFormat = require("../dataset/date_format");

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
 * @return {tf.Model} A compiled model instance.
 */
function createModel(
  inputVocabSize,
  outputVocabSize,
  inputLength,
  outputLength
) {
  const embeddingDims = 64;
  const lstmUnits = 64;

  /** ENCODER */
  const encoderEmbeddingInput = tf.input({ shape: [inputLength] });

  // Select the embedding vectors by providing the `encoderEmbeddingInput` with the indices
  let encoderEmbeddingOutput = tf.layers
    .embedding({
      inputDim: inputVocabSize,
      outputDim: embeddingDims,
      inputLength,
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
  const decoderEmbeddingInput = tf.input({ shape: [outputLength] });
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
  // (W1 * e_j + W2 * d_i)
  let attention = tf.layers
    .dot({ axes: [2, 2], name: "attentionDot" })
    .apply([decoderLSTMOutput, encoderLSTMOutput]);
  // Apply soft max activation to map values into probabilities
  // This produces the Attention Weights
  attention = tf.layers
    .activation({ activation: "softmax", name: "attentionSoftMax" })
    .apply(attention);

  // Apply "Attention Weights" on each output state of encoder LSTM
  // The result "shows" where (at what part of input) to "focus" on
  // The result is the Context Vector
  const context = tf.layers
    .dot({ axes: [2, 1], name: "context" })
    .apply([attention, encoderLSTMOutput]);

  // Concatenates the decoder LSTM Output with the Context Vector
  const decoderCombinedContext = tf.layers
    .concatenate({ name: "combinedContext" })
    .apply([context, decoderLSTMOutput]);

  // Apply the same "dense" layer over `time` dimension.
  // The input should be at least 3D, and the dimension of the index `1` will be considered to be the temporal dimension.
  // More info: https://js.tensorflow.org/api/latest/#layers.timeDistributed
  // Input is [null, 10, 128] "null" (sample size) is not considered.
  // This means that the input shape is [10, 128]. Index `1` is 128. So the timeDistributed will be applied over 128
  // Outputs is [null, 10, 64] lstmUnits = 64
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

/**
 * Perform sequence-to-sequence decoding for date conversion.
 *
 * @param {tf.Model} model The model to be used for the sequence-to-sequence
 *   decoding, with two inputs:
 *   1. Encoder input of shape `[numExamples, inputLength]`
 *   2. Decoder input of shape `[numExamples, outputLength]`
 *   and one output:
 *   1. Decoder softmax probability output of shape
 *      `[numExamples, outputLength, outputVocabularySize]`
 * @param {string} inputStr Input date string to be converted.
 * @return {{outputStr: string, attention?: tf.Tensor}}
 *   - The `outputStr` field is the output date string.
 *   - If and only if `getAttention` is `true`, the `attention` field will
 *     be populated by attention matrix as a `tf.Tensor` of
 *     dtype `float32` and shape `[]`.
 */
async function runSeq2SeqInference(model, inputStr, getAttention = false) {
  return tf.tidy(() => {
    const encoderInput = dateFormat.encodeInputDateStrings([inputStr]);
    const decoderInput = tf.buffer([1, dateFormat.OUTPUT_LENGTH]);
    decoderInput.set(dateFormat.START_CODE, 0, 0);

    for (let i = 1; i < dateFormat.OUTPUT_LENGTH; ++i) {
      const predictOut = model.predict([encoderInput, decoderInput.toTensor()]);

      const output = predictOut.argMax(2).dataSync()[i - 1];
      predictOut.dispose();
      decoderInput.set(output, 0, i);
    }

    const output = { outputStr: "" };

    // The `tf.Model` instance used for the final time step varies depending on
    // whether the attention matrix is requested or not.
    let finalStepModel = model;
    if (getAttention) {
      // If the attention matrix is requested, construct a two-output model.
      // - The 1st output is the original decoder output.
      // - The 2nd output is the attention matrix.
      finalStepModel = tf.model({
        inputs: model.inputs,
        outputs: model.outputs.concat([model.getLayer("attention").output])
      });
    }

    const finalPredictOut = finalStepModel.predict([
      encoderInput,
      decoderInput.toTensor()
    ]);

    let decoderFinalOutput; // The decoder's final output.
    if (getAttention) {
      decoderFinalOutput = finalPredictOut[0];
      output.attention = finalPredictOut[1];
    } else {
      decoderFinalOutput = finalPredictOut;
    }
    decoderFinalOutput = decoderFinalOutput.argMax(2).dataSync()[
      dateFormat.OUTPUT_LENGTH - 1
    ];

    for (let i = 1; i < decoderInput.shape[1]; ++i) {
      output.outputStr += dateFormat.OUTPUT_VOCAB[decoderInput.get(0, i)];
    }
    output.outputStr += dateFormat.OUTPUT_VOCAB[decoderFinalOutput];
    return output;
  });
}

module.exports = {
  createModel,
  runSeq2SeqInference
};
