/**
 * Bahdanau's Attention Paper: https://arxiv.org/abs/1409.0473
 *
 * More info: https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05
 */
const tf = require("@tensorflow/tfjs");

/**
 * @typedef {import('@tensorflow/tfjs').Tensor} Tensor
 */

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
  const lstmUnits = 32;

  let Tx = inputLength;
  let Ty = outputLength;
  let ns = lstmUnits * 2; //
  let na = lstmUnits;
  let humanVocabSize = inputVocabSize;
  let machineVocabSize = outputVocabSize;

  const { static, postActivationLSTMCell, outputLayer } = initConstLayers(
    ns,
    Tx,
    outputVocabSize
  );

  let X = tf.input({ shape: [Tx, humanVocabSize] });
  let s0 = tf.input({ shape: [ns], name: "s0" });
  let c0 = tf.input({ shape: [ns], name: "c0" });
  let s = s0;
  let c = s;

  let outputs = [];

  let a = tf.layers
    .bidirectional({
      layer: tf.layers.lstm({
        units: na,
        returnSequences: true
      })
    })
    .apply(X);

  for (let t = 0; t < Ty; t++) {
    let context = oneStepAttention(a, s, static);

    let res = postActivationLSTMCell.apply(context, { initialState: [s, c] });
    s = res[0];
    c = res[1];

    let out = outputLayer.apply(s);

    outputs.push(out);
  }

  let model = tf.model({
    inputs: [X, s0, c0],
    outputs: outputs
  });

  model.compile({ loss: "categoricalCrossentropy", optimizer: "adam" });
  // return 1;
  return model;
}

function initConstLayers(lstmUnits, inputLength, outputVocabSize) {
  // Layers shared across one time step
  const static = {
    repeator: tf.layers.repeatVector({ n: inputLength }),
    concatenator: tf.layers.concatenate({ axis: -1 }),
    densor1: tf.layers.dense({ units: 10, activation: "tanh" }),
    densor2: tf.layers.dense({ units: 1, activation: "relu" }),
    activator: tf.layers.activation({
      activation: "softmax",
      name: "attention_weights"
    }), // We are using a custom softmax(axis = 1) loaded in this notebook
    dotor: tf.layers.dot({ axes: 1 })
  };

  const postActivationLSTMCell = tf.layers.lstm({
    units: lstmUnits,
    returnState: true
  });

  const outputLayer = tf.layers.dense({
    activation: "softmax",
    units: outputVocabSize
  });

  return { static, postActivationLSTMCell, outputLayer };
}

/**
 * @param {Tensor} a Encoder's Hidden States
 * @param {Tensor} s_prev Decoder's previous hidden state
 * @param {Object} static Static layers
 */
function oneStepAttention(a, s_prev, static) {
  s_prev = static.repeator.apply(s_prev);

  let concat = static.concatenator.apply([a, s_prev]);

  let e = static.densor1.apply(concat);

  let energies = static.densor2.apply(e);

  let alphas = static.activator.apply(energies);

  let context = static.dotor.apply([alphas, a]);

  return context;
}

module.exports = {
  oneStepAttention,
  createModel
};
