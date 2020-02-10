const tf = require("@tensorflow/tfjs");
const AttentionBahdanau = require("./attention");

/**
 * Bahdanau's Decoder
 */
class DecoderBahdanau extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  call(input) {
    const encodersHidden = input[0];
    const something = input[1];

    return input;
  }

  static get className() {
    return "Decoder";
  }
}

module.exports = DecoderBahdanau;
