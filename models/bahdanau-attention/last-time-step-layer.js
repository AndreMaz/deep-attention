const tf = require("@tensorflow/tfjs");

/**
 * A custom layer used to obtain the last time step of an RNN sequential
 * output.
 */
class GetLastTimestepLayer extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.supportMasking = true;
  }

  computeOutputShape(inputShape) {
    const outputShape = inputShape.slice();
    outputShape.splice(outputShape.length - 2, 1);
    return outputShape;
  }

  call(input) {
    if (Array.isArray(input)) {
      input = input[0];
    }
    const inputRank = input.shape.length;
    tf.util.assert(inputRank === 3, `Invalid input rank: ${inputRank}`);
    return input.gather([input.shape[1] - 1], 1).squeeze([1]);
  }

  static get className() {
    return "GetLastTimestepLayer";
  }
}

module.exports = GetLastTimestepLayer;
