const tf = require("@tensorflow/tfjs");

/**
 * @typedef {import('@tensorflow/tfjs').Tensor} Tensor
 */

/**
 * Bahdanau's Decoder
 */
class EncoderBahdanau extends tf.layers.Layer {
  constructor(config) {
    super(config || {});

    this.inputVocabSize = config.inputVocabSize;
    this.embeddingDims = config.embeddingDims;
    this.inputLength = config.inputLength;
    this.lstmUnits = config.lstmUnits;
    this.batchSize = config.batchSize;

    // Selects rows from embedding by `encoderInput` values
    this.embedding = tf.layers.embedding({
      inputDim: this.inputVocabSize,
      outputDim: this.embeddingDims,
      inputLength: this.inputLength,
      maskZero: true
    });

    this.LSTM = tf.layers.lstm({
      units: this.lstmUnits,
      returnSequences: true,
      returnState: true
    });
  }

  computeOutputShape(inputShape) {
    return [null, this.inputLength, this.lstm];
  }

  /**
   * @param {Tensor} input Tensor with indices
   */
  call(input) {
    return tf.tidy(() => {
      /** @type {Tensor} Decoder's Input */
      let y = input[0];
      /** @type {Tensor} Decoder's Last Hidden State */
      const hidden = input[1];

      const encoderOutput = this.embedding.apply(y);

      const lstmOutput = this.LSTM.apply(encoderOutput, {
        // initialState: [hidden, hidden]
      });

      return { output: lstmOutput[0], lastState: lstmOutput[1] };
    });
  }

  initHiddenState() {
    return tf.zeros([this.batchSize, this.lstmUnits]);
  }

  static get className() {
    return "Attention";
  }
}

tf.serialization.registerClass(EncoderBahdanau);
module.exports = EncoderBahdanau;
