/**
 * Bahdanau's Attention Paper: https://arxiv.org/abs/1409.0473
 *
 * More info: https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05
 */
const tf = require("@tensorflow/tfjs");

/**
 * @typedef {import('@tensorflow/tfjs').Tensor} Tensor
 */

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

  // Layers shared across one time step
  let static = {
    repeator: tf.layers.repeatVector({ n: inputVocabSize }),
    concatenator: tf.layers.concatenate({ axis: -1 }),
    densor1: tf.layers.dense(10, { activation: "tanh" }),
    densor2: tf.layers.dense(1, { activation: "relu" }),
    activator: tf.layers.activation(softmax, { name: "attention_weights" }), // We are using a custom softmax(axis = 1) loaded in this notebook
    dotor: tf.layers.dot({ axes: 1 })
  };

  return model;
}

/**
 * @param {Tensor} a Encoder's Hidden States
 * @param {Tensor} s_prev Decoder's previous hidden state
 * @param {Object} static Static layers
 */
function oneStepAttention(a, s_prev, static) {
  s_prev = static.repeator.apply(s_prev);

  let concat = static.repeator.concatenator.apply([a, s_prev]);

  let e = static.densor1.apply(concat);

  let energies = static.densor2.apply(e);

  let alphas = static.activator.apply(energies);

  let context = static.dotor.apply([alphas, a]);

  return context;
}

module.exports = {
  createModel
};
